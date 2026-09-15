from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services.ingestion.supabase_video import SupabaseVideoStore
from services.lab_contract import LAB_DEFAULTS, lab_defaults, resolve_jersey_backend
from services.supabase_service import SupabaseRunStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_ROOT = PROJECT_ROOT / "evaluation" / "worker_jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = max(1, int(os.getenv("WORKER_MAX_JOBS", "1")))

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="basketball-worker")
_state_lock = threading.Lock()
_control_lock = threading.Lock()
_job_futures: dict[str, Future[Any]] = {}
_job_processes: dict[str, subprocess.Popen[str]] = {}
_cancel_requested: set[str] = set()
_supabase_store = SupabaseRunStore()

app = FastAPI(
    title="CourtFlux Worker",
    version="0.7.0",
    description="CourtFlux analysis worker using the canonical Beta Lab analysis contract.",
)


class JobRequest(BaseModel):
    video_source: str = Field(default=LAB_DEFAULTS["video_source"], min_length=1, max_length=2048)
    start_time_sec: float = Field(default=0.0, ge=0.0)
    end_time_sec: float = Field(default=60.0, gt=0.0)
    model: str = Field(default=LAB_DEFAULTS["model"], min_length=1, max_length=512)
    detector_conf: float = Field(default=LAB_DEFAULTS["detector_conf"], ge=0.01, le=0.95)
    inference_conf: float = Field(default=LAB_DEFAULTS["inference_conf"], ge=0.001, le=0.95)
    ball_conf: float = Field(default=LAB_DEFAULTS["ball_conf"], ge=0.001, le=0.95)
    imgsz: int = Field(default=LAB_DEFAULTS["imgsz"], ge=320, le=1600)
    device: Literal["mps", "cpu", "cuda"] = LAB_DEFAULTS["device"]
    tracker: Literal["botsort.yaml", "bytetrack.yaml"] = LAB_DEFAULTS["tracker"]
    match_config: dict[str, Any]
    detection_only: bool = LAB_DEFAULTS["detection_only"]
    enable_jersey: bool = LAB_DEFAULTS["enable_jersey"]
    jersey_backend: Literal[
        "best_temporal",
        "roboflow_classification",
        "smolvlm2_jersey",
        "trocr_small_printed",
        "parseq_soccernet",
        "ebqwen_vl",
        "uncertainty_vit_b",
        "legacy_digit_heads",
        "auto",
    ] = LAB_DEFAULTS["jersey_backend"]
    jersey_interval_frames: int = Field(default=LAB_DEFAULTS["jersey_interval_frames"], ge=1, le=300)
    persist_results: bool = LAB_DEFAULTS["persist_results"]
    live_preview: bool = LAB_DEFAULTS["live_preview"]
    preview_interval_frames: int = Field(default=LAB_DEFAULTS["preview_interval_frames"], ge=1, le=300)
    single_forward: bool = LAB_DEFAULTS["single_forward"]
    smart_ball_mode: Literal["off", "verify", "zoom", "full"] = LAB_DEFAULTS["smart_ball_mode"]
    smart_ball_crop_imgsz: int = Field(default=LAB_DEFAULTS["smart_ball_crop_imgsz"], ge=320, le=1600)
    db_batch_size: int = Field(default=100, ge=1, le=5000)
    diagnostics_batch_size: int = Field(default=100, ge=1, le=5000)
    platform_run_id: str | None = Field(default=None, max_length=36)
    platform_job_id: str | None = Field(default=None, max_length=36)
    workspace_id: str | None = Field(default=None, max_length=36)
    user_id: str | None = Field(default=None, max_length=36)
    subscription_plan: str | None = Field(default=None, max_length=64)


class JobCreated(BaseModel):
    job_id: str
    status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_uuid(value: str | None, field_name: str) -> str | None:
    if not value:
        return None
    try:
        return str(uuid.UUID(value))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{field_name} must be a valid UUID") from exc


def require_worker_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("WORKER_API_TOKEN", "").strip()
    if not expected:
        host = os.getenv("WORKER_HOST", "127.0.0.1").strip().lower()
        if host in {"127.0.0.1", "localhost", "::1"}:
            return
        raise HTTPException(status_code=503, detail="WORKER_API_TOKEN is required when WORKER_HOST is not loopback")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing worker bearer token")
    if authorization[len("Bearer "):] != expected:
        raise HTTPException(status_code=403, detail="Invalid worker bearer token")


def job_dir(job_id: str) -> Path:
    if not job_id or any(c not in "0123456789abcdef-" for c in job_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid job id")
    return JOBS_ROOT / job_id


def job_state_path(job_id: str) -> Path:
    return job_dir(job_id) / "job.json"


def read_state(job_id: str) -> dict[str, Any]:
    path = job_state_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        artifacts = dict(state.get("artifacts") or {})
        directory = job_dir(job_id)
        artifacts["preview"] = (directory / "latest_preview.jpg").exists()
        artifacts["analysis_video"] = (directory / "analysis_preview.mp4").exists()
        state["artifacts"] = artifacts
        return state
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read job state: {exc}") from exc


def write_state(job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = job_state_path(job_id)
    with _state_lock:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        current.update(updates)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    return current


def validate_match_config(config: dict[str, Any]) -> None:
    metadata = config.get("metadata")
    teams = config.get("teams")
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=422, detail="match_config.metadata must be an object")
    if not isinstance(teams, dict) or len(teams) < 2:
        raise HTTPException(status_code=422, detail="match_config.teams must contain at least two teams")
    home, away = metadata.get("home_team"), metadata.get("away_team")
    if not home or not away:
        raise HTTPException(status_code=422, detail="match_config must define home_team and away_team")
    if home not in teams or away not in teams:
        raise HTTPException(status_code=422, detail="home_team and away_team must exist in match_config.teams")


def cancellation_requested(job_id: str) -> bool:
    with _control_lock:
        return job_id in _cancel_requested


def mark_cancel_requested(job_id: str) -> dict[str, Any]:
    state = read_state(job_id)
    if str(state.get("status") or "") in {"completed", "failed", "canceled"}:
        return state
    with _control_lock:
        _cancel_requested.add(job_id)
    progress = dict(state.get("progress") or {})
    progress.update({"phase": "canceling", "eta_sec": None, "estimated_completion_at": None})
    return write_state(
        job_id,
        {
            "cancel_requested": True,
            "cancel_requested_at": utc_now(),
            "progress": progress,
        },
    )


def finalize_canceled(job_id: str, platform_run_id: str | None, reason: str = "Canceled by user") -> dict[str, Any]:
    state = read_state(job_id)
    status = str(state.get("status") or "")
    if status in {"completed", "failed", "canceled"}:
        return state

    completed_at = utc_now()
    progress = dict(state.get("progress") or {})
    progress.update(
        {
            "phase": "canceled",
            "eta_sec": None,
            "estimated_completion_at": None,
            "last_progress_at": progress.get("last_progress_at") or completed_at,
        }
    )
    updated = write_state(
        job_id,
        {
            "status": "canceled",
            "completed_at": completed_at,
            "cancel_requested": True,
            "cancel_reason": reason,
            "error": None,
            "progress": progress,
        },
    )
    _supabase_store.update_job(
        job_id,
        {"status": "canceled", "completed_at": completed_at, "error": None},
    )
    _supabase_store.update_run(
        platform_run_id,
        {"status": "canceled", "completed_at": completed_at},
    )
    return updated


def _signal_process_group(process: subprocess.Popen[str], sig: int) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, sig)
            return
        except ProcessLookupError:
            return
        except Exception:
            pass
    process.send_signal(sig)


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        _signal_process_group(process, signal.SIGINT)
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    try:
        if os.name != "nt":
            _signal_process_group(process, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    try:
        if os.name != "nt":
            _signal_process_group(process, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=2)
    except Exception:
        pass


def _register_future(job_id: str, future: Future[Any]) -> None:
    with _control_lock:
        _job_futures[job_id] = future

    def cleanup(done_future: Future[Any]) -> None:
        with _control_lock:
            if _job_futures.get(job_id) is done_future:
                _job_futures.pop(job_id, None)

    future.add_done_callback(cleanup)


def run_job(job_id: str, request_data: dict[str, Any]) -> None:
    directory = job_dir(job_id)
    log_path = directory / "console.log"
    match_config_path = directory / "match_config.json"
    analyzer = PROJECT_ROOT / "services" / "analytics" / "analyzer.py"
    platform_run_id = request_data.get("platform_run_id")
    preview_path = directory / "latest_preview.jpg"
    preview_video_path = directory / "analysis_preview.mp4"
    total_window_sec = max(0.0, float(request_data["end_time_sec"]) - float(request_data["start_time_sec"]))
    job_started_wall = time.monotonic()
    progress_anchor_wall: float | None = None
    progress_anchor_processed: float | None = None
    process: subprocess.Popen[str] | None = None

    if cancellation_requested(job_id):
        finalize_canceled(job_id, platform_run_id)
        return

    write_state(
        job_id,
        {
            "status": "running",
            "started_at": utc_now(),
            "progress": {
                "phase": "preparing",
                "percent": 0.0,
                "processed_seconds": 0.0,
                "total_seconds": total_window_sec,
                "frames_processed": 0,
                "elapsed_sec": 0.0,
                "eta_sec": None,
                "estimated_completion_at": None,
                "last_progress_at": None,
            },
        },
    )
    _supabase_store.mark_running(job_id, platform_run_id)
    summary_path: Path | None = None
    recent_lines: list[str] = []

    try:
        if cancellation_requested(job_id):
            finalize_canceled(job_id, platform_run_id)
            return

        video_source = str(request_data["video_source"])
        if video_source.startswith("supabase://"):
            video_store = SupabaseVideoStore(output_dir=str(PROJECT_ROOT / "data" / "uploads"))
            video_source = video_store.get_local_video(
                video_source,
                expected_workspace_id=request_data.get("workspace_id"),
            )

        if cancellation_requested(job_id):
            finalize_canceled(job_id, platform_run_id)
            return

        command = [
            sys.executable,
            str(analyzer),
            "--video",
            video_source,
            "--start",
            str(request_data["start_time_sec"]),
            "--end",
            str(request_data["end_time_sec"]),
            "--model",
            request_data["model"],
            "--conf",
            str(request_data["detector_conf"]),
            "--imgsz",
            str(request_data["imgsz"]),
            "--device",
            request_data["device"],
            "--tracker",
            request_data["tracker"],
            "--match-config",
            str(match_config_path),
            "--jersey-interval",
            str(request_data["jersey_interval_frames"]),
            "--diagnostics-batch-size",
            str(request_data["diagnostics_batch_size"]),
        ]
        if request_data.get("detection_only", True):
            command.append("--detection-only")
        if not request_data.get("enable_jersey", False):
            command.append("--no-jersey")
        if not request_data.get("persist_results", True):
            command.append("--no-persist")

        selected_backend = resolve_jersey_backend(
            str(request_data.get("jersey_backend") or LAB_DEFAULTS["jersey_backend"])
        )
        hf_home = Path(
            os.getenv("HF_HOME", str(PROJECT_ROOT / "weights" / "huggingface"))
        ).expanduser().resolve()
        hf_home.mkdir(parents=True, exist_ok=True)

        child_env = os.environ.copy()
        child_env["BASKETBALL_JERSEY_BACKEND"] = selected_backend
        child_env["HF_HOME"] = str(hf_home)
        child_env["BASKETBALL_INFERENCE_CONF"] = str(request_data["inference_conf"])
        child_env["BASKETBALL_BALL_CONF"] = str(request_data["ball_conf"])
        child_env["BASKETBALL_SINGLE_FORWARD"] = "1" if request_data.get("single_forward", True) else "0"
        child_env["BASKETBALL_SMART_BALL_MODE"] = str(request_data.get("smart_ball_mode", "off"))
        child_env["BASKETBALL_BALL_CROP_IMGSZ"] = str(request_data.get("smart_ball_crop_imgsz", 640))
        if request_data.get("live_preview", True) and request_data.get("detection_only", True):
            child_env["BASKETBALL_PREVIEW_PATH"] = str(preview_path)
            child_env["BASKETBALL_PREVIEW_VIDEO_PATH"] = str(preview_video_path)
            child_env["BASKETBALL_PREVIEW_MATCH_CONFIG"] = str(match_config_path)
            child_env["BASKETBALL_PREVIEW_INTERVAL"] = str(request_data["preview_interval_frames"])

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(
                f"[WORKER] jersey_backend={selected_backend} hf_home={hf_home} persistence=supabase_only\n"
            )
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=os.name != "nt",
            )
            with _control_lock:
                _job_processes[job_id] = process

            if cancellation_requested(job_id):
                try:
                    _signal_process_group(process, signal.SIGINT)
                except Exception:
                    pass

            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                text = line.rstrip()
                recent_lines.append(text)
                recent_lines = recent_lines[-40:]

                if text.startswith("[PROGRESS]"):
                    try:
                        payload = json.loads(text.replace("[PROGRESS]", "", 1).strip())
                        processed = max(0.0, float(payload.get("processed_seconds", 0.0) or 0.0))
                        total = max(0.0, float(payload.get("total_seconds", total_window_sec) or total_window_sec))
                        raw_percent = max(0.0, min(100.0, float(payload.get("percent", 0.0) or 0.0)))
                        frames_processed = max(0, int(payload.get("frames_processed", 0) or 0))
                        now_wall = time.monotonic()
                        elapsed_sec = max(0.0, now_wall - job_started_wall)

                        if progress_anchor_wall is None:
                            progress_anchor_wall = now_wall
                            progress_anchor_processed = processed

                        eta_sec: float | None = None
                        if progress_anchor_wall is not None and progress_anchor_processed is not None:
                            tracking_wall = max(0.0, now_wall - progress_anchor_wall)
                            tracking_video = max(0.0, processed - progress_anchor_processed)
                            if tracking_wall >= 3.0 and tracking_video >= 0.25 and processed < total:
                                seconds_per_video_second = tracking_wall / tracking_video
                                eta_sec = max(0.0, (total - processed) * seconds_per_video_second)

                        phase = "finalizing" if raw_percent >= 99.5 else "tracking"
                        display_percent = 99.0 if phase == "finalizing" else raw_percent
                        estimated_completion_at = (
                            (datetime.now(timezone.utc) + timedelta(seconds=eta_sec)).isoformat()
                            if eta_sec is not None
                            else None
                        )
                        is_canceling = cancellation_requested(job_id)
                        write_state(
                            job_id,
                            {
                                "progress": {
                                    "phase": "canceling" if is_canceling else phase,
                                    "percent": round(display_percent, 2),
                                    "processed_seconds": round(processed, 3),
                                    "total_seconds": round(total, 3),
                                    "frames_processed": frames_processed,
                                    "elapsed_sec": round(elapsed_sec, 2),
                                    "eta_sec": None if is_canceling else round(eta_sec, 2) if eta_sec is not None else None,
                                    "estimated_completion_at": None if is_canceling else estimated_completion_at,
                                    "last_progress_at": utc_now(),
                                }
                            },
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass

                if text.startswith("[SUMMARY]"):
                    candidate = Path(text.replace("[SUMMARY]", "", 1).strip())
                    if candidate.exists():
                        summary_path = candidate

            return_code = process.wait()

        if cancellation_requested(job_id):
            finalize_canceled(job_id, platform_run_id)
            return

        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path and summary_path.exists() else None
        artifacts = {
            "preview": preview_path.exists(),
            "analysis_video": preview_video_path.exists(),
        }
        if return_code == 0:
            write_state(
                job_id,
                {
                    "status": "completed",
                    "completed_at": utc_now(),
                    "return_code": return_code,
                    "summary": summary,
                    "artifacts": artifacts,
                    "progress": {
                        "phase": "completed",
                        "percent": 100.0,
                        "processed_seconds": total_window_sec,
                        "total_seconds": total_window_sec,
                        "frames_processed": (read_state(job_id).get("progress") or {}).get("frames_processed", 0),
                        "elapsed_sec": round(max(0.0, time.monotonic() - job_started_wall), 2),
                        "eta_sec": 0.0,
                        "estimated_completion_at": utc_now(),
                        "last_progress_at": utc_now(),
                    },
                },
            )
            _supabase_store.mark_completed(job_id, platform_run_id, summary, request_data)
        else:
            error_tail = "\n".join(recent_lines[-12:])
            previous_progress = dict((read_state(job_id).get("progress") or {}))
            previous_progress.update({"phase": "failed", "eta_sec": None, "estimated_completion_at": None})
            write_state(
                job_id,
                {
                    "status": "failed",
                    "completed_at": utc_now(),
                    "return_code": return_code,
                    "summary": summary,
                    "error": error_tail,
                    "artifacts": artifacts,
                    "progress": previous_progress,
                },
            )
            _supabase_store.mark_failed(job_id, platform_run_id, error_tail, error_tail)
    except BaseException as exc:
        if cancellation_requested(job_id):
            finalize_canceled(job_id, platform_run_id)
            return
        error_text = f"{type(exc).__name__}: {exc}"
        try:
            previous_progress = dict((read_state(job_id).get("progress") or {}))
        except Exception:
            previous_progress = {}
        previous_progress.update({"phase": "failed", "eta_sec": None, "estimated_completion_at": None})
        write_state(
            job_id,
            {"status": "failed", "completed_at": utc_now(), "error": error_text, "progress": previous_progress},
        )
        _supabase_store.mark_failed(
            job_id,
            platform_run_id,
            error_text,
            "\n".join(recent_lines[-12:]) if recent_lines else None,
        )
    finally:
        with _control_lock:
            if process is not None and _job_processes.get(job_id) is process:
                _job_processes.pop(job_id, None)
            _cancel_requested.discard(job_id)


@app.get("/health", dependencies=[Depends(require_worker_token)])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "worker": "courtflux",
        "max_parallel_jobs": MAX_WORKERS,
        "supabase_persistence": _supabase_store.enabled,
        "lab_defaults": lab_defaults(),
        "devices": {
            "mps": bool(torch.backends.mps.is_available()),
            "cuda": bool(torch.cuda.is_available()),
            "cpu": True,
        },
    }


@app.post("/jobs", response_model=JobCreated, dependencies=[Depends(require_worker_token)])
def create_job(payload: JobRequest) -> JobCreated:
    if payload.end_time_sec <= payload.start_time_sec:
        raise HTTPException(status_code=422, detail="end_time_sec must be greater than start_time_sec")
    if payload.inference_conf > min(payload.detector_conf, payload.ball_conf):
        raise HTTPException(status_code=422, detail="inference_conf must be <= both detector_conf and ball_conf")
    validate_match_config(payload.match_config)
    platform_job_id = normalize_uuid(payload.platform_job_id, "platform_job_id")
    normalize_uuid(payload.platform_run_id, "platform_run_id")
    normalize_uuid(payload.workspace_id, "workspace_id")
    normalize_uuid(payload.user_id, "user_id")
    job_id = platform_job_id or str(uuid.uuid4())
    directory = job_dir(job_id)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="Job already exists") from exc

    request_data = payload.model_dump()
    # Worker-backed runs are Supabase-only. Never allow an older caller to revive
    # the legacy local tracking.db path.
    request_data["persist_results"] = False
    request_data["jersey_backend"] = resolve_jersey_backend(
        str(request_data.get("jersey_backend") or LAB_DEFAULTS["jersey_backend"])
    )
    (directory / "request.json").write_text(json.dumps(request_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "match_config.json").write_text(
        json.dumps(payload.match_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_state(
        job_id,
        {
            "job_id": job_id,
            "platform_run_id": payload.platform_run_id,
            "workspace_id": payload.workspace_id,
            "status": "queued",
            "created_at": utc_now(),
            "request": {k: v for k, v in request_data.items() if k != "match_config"},
        },
    )
    _supabase_store.update_job(
        job_id,
        {
            "status": "queued",
            "worker_id": os.getenv("COURTFLUX_WORKER_ID", os.getenv("HOSTNAME", "local-worker")),
        },
    )
    future = _executor.submit(run_job, job_id, request_data)
    _register_future(job_id, future)
    return JobCreated(job_id=job_id, status="queued")


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_worker_token)])
def cancel_job(job_id: str) -> dict[str, Any]:
    state = read_state(job_id)
    status = str(state.get("status") or "")
    platform_run_id = state.get("platform_run_id")

    if status == "canceled":
        return {"job_id": job_id, "status": "canceled", "already_terminal": True}
    if status in {"completed", "failed"}:
        raise HTTPException(status_code=409, detail=f"Analysis is already {status}")
    if status not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail=f"Analysis cannot be canceled from status {status or 'unknown'}")

    requested_state = mark_cancel_requested(job_id)
    requested_status = str(requested_state.get("status") or "")
    if requested_status in {"completed", "failed", "canceled"}:
        if requested_status == "canceled":
            return {"job_id": job_id, "status": "canceled", "already_terminal": True}
        raise HTTPException(status_code=409, detail=f"Analysis is already {requested_status}")

    with _control_lock:
        future = _job_futures.get(job_id)
        process = _job_processes.get(job_id)

    if future is not None and future.cancel():
        canceled = finalize_canceled(job_id, platform_run_id)
        with _control_lock:
            _cancel_requested.discard(job_id)
        return {"job_id": job_id, "status": canceled.get("status", "canceled")}

    if process is not None:
        stop_process(process)
        canceled = finalize_canceled(job_id, platform_run_id)
        return {"job_id": job_id, "status": canceled.get("status", "canceled")}

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        time.sleep(0.05)
        current = read_state(job_id)
        current_status = str(current.get("status") or "")
        if current_status in {"completed", "failed", "canceled"}:
            with _control_lock:
                _cancel_requested.discard(job_id)
            return {"job_id": job_id, "status": current_status}
        with _control_lock:
            process = _job_processes.get(job_id)
        if process is not None:
            stop_process(process)
            canceled = finalize_canceled(job_id, platform_run_id)
            return {"job_id": job_id, "status": canceled.get("status", "canceled")}

    return {"job_id": job_id, "status": "cancel_requested"}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_worker_token)])
def get_job(job_id: str) -> dict[str, Any]:
    return read_state(job_id)


@app.get("/jobs/{job_id}/log", dependencies=[Depends(require_worker_token)])
def get_job_log(job_id: str, tail: int = 120) -> dict[str, Any]:
    tail = max(1, min(tail, 500))
    path = job_dir(job_id) / "console.log"
    if not path.exists():
        return {"job_id": job_id, "lines": []}
    return {
        "job_id": job_id,
        "lines": path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:],
    }


@app.get("/jobs/{job_id}/preview", dependencies=[Depends(require_worker_token)])
def get_job_preview(job_id: str):
    path = job_dir(job_id) / "latest_preview.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not available yet")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/jobs/{job_id}/preview-video", dependencies=[Depends(require_worker_token)])
def get_job_preview_video(job_id: str):
    path = job_dir(job_id) / "analysis_preview.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis preview video not available")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'inline; filename="analysis_preview.mp4"',
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.worker_api:app",
        host=os.getenv("WORKER_HOST", "127.0.0.1"),
        port=int(os.getenv("WORKER_PORT", "8787")),
        reload=False,
    )
