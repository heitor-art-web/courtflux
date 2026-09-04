from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JOBS_ROOT = PROJECT_ROOT / "evaluation" / "worker_jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_VIDEO = "https://www.youtube.com/watch?v=OwiQgAMqyUs"
DEFAULT_MODEL = "weights/Model_v0.1.pt"
MAX_WORKERS = max(1, int(os.getenv("WORKER_MAX_JOBS", "1")))

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="basketball-worker")
_state_lock = threading.Lock()

app = FastAPI(
    title="Basketball AI Analytics Worker",
    version="0.2.0",
    description="Authenticated worker for Vercel beta and performance runs.",
)


class JobRequest(BaseModel):
    video_source: str = Field(default=DEFAULT_VIDEO, min_length=1, max_length=2048)
    start_time_sec: float = Field(default=0.0, ge=0.0)
    end_time_sec: float = Field(default=60.0, gt=0.0)
    model: str = Field(default=DEFAULT_MODEL, min_length=1, max_length=512)
    detector_conf: float = Field(default=0.10, ge=0.01, le=0.95)
    imgsz: int = Field(default=640, ge=320, le=1600)
    device: Literal["mps", "cpu", "cuda"] = "mps"
    tracker: Literal["botsort.yaml", "bytetrack.yaml"] = "botsort.yaml"
    match_config: dict[str, Any]
    detection_only: bool = True
    enable_jersey: bool = False
    jersey_interval_frames: int = Field(default=12, ge=1, le=300)
    persist_results: bool = True
    db_batch_size: int = Field(default=100, ge=1, le=5000)
    diagnostics_batch_size: int = Field(default=100, ge=1, le=5000)


class JobCreated(BaseModel):
    job_id: str
    status: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_worker_token(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("WORKER_API_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="WORKER_API_TOKEN is not configured on the worker")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Missing worker bearer token")
    supplied = authorization[len(prefix):]
    if supplied != expected:
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
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read job state: {exc}") from exc


def write_state(job_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = job_state_path(job_id)
    with _state_lock:
        current: dict[str, Any] = {}
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
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
    home = metadata.get("home_team")
    away = metadata.get("away_team")
    if not home or not away:
        raise HTTPException(status_code=422, detail="match_config must define home_team and away_team")
    if home not in teams or away not in teams:
        raise HTTPException(status_code=422, detail="home_team and away_team must exist in match_config.teams")


def run_job(job_id: str, request_data: dict[str, Any]) -> None:
    directory = job_dir(job_id)
    log_path = directory / "console.log"
    match_config_path = directory / "match_config.json"
    analyzer = PROJECT_ROOT / "services" / "analytics" / "analyzer.py"
    write_state(job_id, {"status": "running", "started_at": utc_now()})

    command = [
        sys.executable,
        str(analyzer),
        "--video", request_data["video_source"],
        "--start", str(request_data["start_time_sec"]),
        "--end", str(request_data["end_time_sec"]),
        "--model", request_data["model"],
        "--conf", str(request_data["detector_conf"]),
        "--imgsz", str(request_data["imgsz"]),
        "--device", request_data["device"],
        "--tracker", request_data["tracker"],
        "--match-config", str(match_config_path),
        "--jersey-interval", str(request_data["jersey_interval_frames"]),
        "--db-batch-size", str(request_data["db_batch_size"]),
        "--diagnostics-batch-size", str(request_data["diagnostics_batch_size"]),
        "--reset-db",
    ]
    if request_data.get("detection_only", True):
        command.append("--detection-only")
    if not request_data.get("enable_jersey", False):
        command.append("--no-jersey")
    if not request_data.get("persist_results", True):
        command.append("--no-persist")

    summary_path: Path | None = None
    recent_lines: list[str] = []
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                text = line.rstrip()
                recent_lines.append(text)
                recent_lines = recent_lines[-40:]
                if text.startswith("[SUMMARY]"):
                    candidate = Path(text.replace("[SUMMARY]", "", 1).strip())
                    if candidate.exists():
                        summary_path = candidate
            return_code = process.wait()

        summary: dict[str, Any] | None = None
        if summary_path is not None and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        if return_code == 0:
            write_state(job_id, {
                "status": "completed",
                "completed_at": utc_now(),
                "return_code": return_code,
                "summary": summary,
            })
        else:
            write_state(job_id, {
                "status": "failed",
                "completed_at": utc_now(),
                "return_code": return_code,
                "summary": summary,
                "error": "\n".join(recent_lines[-12:]),
            })
    except Exception as exc:
        write_state(job_id, {
            "status": "failed",
            "completed_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
        })


@app.get("/health", dependencies=[Depends(require_worker_token)])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "worker": "basketball-ai-analytics",
        "max_parallel_jobs": MAX_WORKERS,
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
    validate_match_config(payload.match_config)
    job_id = str(uuid.uuid4())
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    request_data = payload.model_dump()
    (directory / "request.json").write_text(json.dumps(request_data, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "match_config.json").write_text(
        json.dumps(payload.match_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_state(job_id, {
        "job_id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "request": {key: value for key, value in request_data.items() if key != "match_config"},
    })
    _executor.submit(run_job, job_id, request_data)
    return JobCreated(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", dependencies=[Depends(require_worker_token)])
def get_job(job_id: str) -> dict[str, Any]:
    return read_state(job_id)


@app.get("/jobs/{job_id}/log", dependencies=[Depends(require_worker_token)])
def get_job_log(job_id: str, tail: int = 120) -> dict[str, Any]:
    tail = max(1, min(tail, 500))
    path = job_dir(job_id) / "console.log"
    if not path.exists():
        return {"job_id": job_id, "lines": []}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"job_id": job_id, "lines": lines[-tail:]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.worker_api:app",
        host=os.getenv("WORKER_HOST", "127.0.0.1"),
        port=int(os.getenv("WORKER_PORT", "8787")),
        reload=False,
    )
