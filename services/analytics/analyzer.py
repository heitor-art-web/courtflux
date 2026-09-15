from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
import torch
from ultralytics import YOLO

from core.config import settings
from services.database_service import DatabaseService
from services.diagnostics import RunDiagnostics
from services.frame_loader import FrameLoader
from services.gameplay_filter import GameplayFilter
from services.ingestion.youtube_downloader import YouTubeDownloader
from services.persistence import db_setup
from services.pipeline import Pipeline
from services.profiler import PipelineProfiler
from services.video_session import VideoSession
from services.vision.camera_tracker import CameraTracker
from services.vision.identity_v2_detector import PerformanceDetector
from services.vision.jersey_backend_factory import build_jersey_visual
from services.vision.team_only_analyzer import TeamOnlyAnalyzer

DEFAULT_MATCH_CONFIG_PATH = PROJECT_ROOT / "data" / "match_config.json"


def _env_bool(name: str, fallback: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return bool(fallback)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _generate_identity_debug(summary_path: Path, run_dir: Path) -> dict | None:
    try:
        from evaluation.p1_jersey_probe import build_identity_debug

        debug = build_identity_debug(
            summary_path=summary_path,
            output_root=run_dir / "identity_debug",
            max_cases=80,
        )
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        extra = dict(payload.get("extra") or {})
        extra["identity_debug"] = debug
        payload["extra"] = extra
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return debug
    except Exception as exc:
        print(
            f"⚠️ Identity debug artifact generation failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def resolve_device(requested: str | None) -> str:
    if requested:
        requested = requested.lower().strip()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cpu":
            return "cpu"
        print(f"⚠️ Requested device '{requested}' is unavailable; falling back automatically.", flush=True)
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(
    video_source: str,
    start_time: float,
    end_time: float | None,
    model_path: str | None = None,
    detector_conf: float = 0.40,
    imgsz: int = 640,
    device: str | None = None,
    tracker: str = "botsort.yaml",
    reset_db: bool = False,
    match_config_path: str | None = None,
    enable_jersey: bool = True,
    jersey_interval_frames: int = 24,
    db_batch_size: int = 100,
    diagnostics_batch_size: int = 100,
    persist_results: bool = True,
):
    """Run the single canonical CourtFlux P1 -> optional P2 analysis path."""
    resolved_device = resolve_device(device)
    chosen_model = model_path or settings.model
    chosen_match_config = Path(match_config_path) if match_config_path else DEFAULT_MATCH_CONFIG_PATH
    if not chosen_match_config.is_absolute():
        chosen_match_config = PROJECT_ROOT / chosen_match_config
    chosen_match_config = chosen_match_config.resolve()

    diagnostics = RunDiagnostics(
        root=PROJECT_ROOT,
        video_source=video_source,
        start_time=start_time,
        end_time=end_time,
        model_path=chosen_model,
        detector_conf=detector_conf,
        imgsz=imgsz,
        device=resolved_device,
        tracker=tracker,
        enable_jersey=enable_jersey,
        jersey_interval_frames=jersey_interval_frames,
        batch_size=diagnostics_batch_size,
    )
    # RunDiagnostics still supports the historical field for older standalone callers,
    # but the canonical P1->P2 runtime must not emit a retired detection-only mode.
    diagnostics.config.pop("detection_only", None)
    diagnostics._write_meta()
    diagnostics.capture_match_config(chosen_match_config)
    print(f"[RUN_DIR] {diagnostics.run_dir.resolve()}", flush=True)

    conn = None
    session = None
    pipeline = None
    visual = None

    try:
        if not chosen_match_config.exists():
            raise FileNotFoundError(f"Match config not found: {chosen_match_config}")

        db_path = PROJECT_ROOT / "data" / "tracking.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if persist_results:
            if reset_db or not db_path.exists():
                db_setup.initialize_database()
            conn = sqlite3.connect(db_path)
            db_service = DatabaseService(conn, batch_size=db_batch_size)
        else:
            db_service = None

        print(f"🎬 Initializing video source: {video_source}", flush=True)
        if "youtube.com" in video_source or "youtu.be" in video_source:
            video_path = YouTubeDownloader(output_dir="data").get_local_video(video_source)
        else:
            video_path = video_source

        end_time_val = end_time if end_time is not None else float("inf")
        session = VideoSession(video_path=video_path, start_time=start_time, end_time=end_time_val)
        session.open()
        success, first_frame = session.cap.read()
        if not success:
            raise RuntimeError(f"Unable to read frame at {start_time:.1f}s")
        session.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        court_calibration_enabled = _env_bool("BASKETBALL_COURT_CALIBRATION", False)
        if court_calibration_enabled:
            print("📐 Canonical runtime: diagnostic court calibration enabled.", flush=True)
            camera_tracker = CameraTracker(first_frame=first_frame)
            court_polygon = camera_tracker.base_polygon
        else:
            print("⚡ Canonical runtime: image-space CourtState; homography disabled.", flush=True)
            camera_tracker = None
            court_polygon = None

        if enable_jersey:
            visual = build_jersey_visual(
                profiling_path=str(chosen_match_config),
                device=resolved_device,
                project_root=PROJECT_ROOT,
            )
        else:
            print("⚡ Jersey disabled; Identity v2 uses team/track evidence only.", flush=True)
            visual = TeamOnlyAnalyzer(profiling_path=str(chosen_match_config))

        chosen_model_path = Path(chosen_model)
        if not chosen_model_path.is_absolute():
            chosen_model_path = PROJECT_ROOT / chosen_model_path
        if not chosen_model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {chosen_model}. Select a checkpoint available on the worker."
            )

        model = YOLO(str(chosen_model_path))
        model.to(resolved_device)
        detector = PerformanceDetector(
            model=model,
            visual_analyzer=visual,
            profiling_path=str(chosen_match_config),
            device=resolved_device,
            detector_conf=detector_conf,
            imgsz=imgsz,
            iou=0.5,
            tracker=tracker,
            max_det=40,
            enable_jersey=enable_jersey,
            jersey_interval_frames=jersey_interval_frames,
        )

        profiler = PipelineProfiler()
        pipeline = Pipeline(
            gameplay_filter=GameplayFilter(
                min_court_ratio=0.15,
                min_motion_ratio=0.003,
                max_grace_frames=15,
                processing_scale=0.5,
            ),
            detector=detector,
            camera_tracker=camera_tracker,
            backboard_roi=None,
            database=db_service,
            renderer=None,
            profiler=profiler,
            db_conn=conn,
            persist_results=persist_results,
        )

        loader = FrameLoader(session)
        frame_id = 0
        total_window_sec = (
            max(0.0, float(end_time_val) - float(start_time))
            if end_time_val != float("inf")
            else None
        )
        while True:
            frame_state = loader.next()
            if frame_state is None or frame_state.timestamp > end_time_val:
                break
            frame_state.frame_id = frame_id
            if court_polygon is not None:
                frame_state.court_polygon = court_polygon
            frame_state = pipeline.process(frame_state)
            diagnostics.log_frame(frame_state)
            frame_id += 1

            if total_window_sec and (frame_id == 1 or frame_id % 15 == 0):
                current_timestamp = float(getattr(frame_state, "timestamp", start_time) or start_time)
                processed_sec = max(
                    0.0,
                    min(current_timestamp, float(end_time_val)) - float(start_time),
                )
                percent = min(100.0, max(0.0, (processed_sec / total_window_sec) * 100.0))
                print(
                    "[PROGRESS] "
                    + json.dumps(
                        {
                            "timestamp_sec": round(current_timestamp, 3),
                            "processed_seconds": round(processed_sec, 3),
                            "total_seconds": round(total_window_sec, 3),
                            "percent": round(percent, 2),
                            "frames_processed": frame_id,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

        pipeline.flush()
        video_fps = float(session.fps or 30.0)
        performance = profiler.snapshot(video_fps=video_fps)
        profiler.report(video_fps=video_fps)

        extra = {
            "status": "completed",
            "video_path": str(video_path),
            "mode": "canonical_p1_p2",
            "p2_enabled": _env_bool("BASKETBALL_P2_ENABLED", False),
            "persist_results": persist_results,
            "court_calibration_enabled": bool(camera_tracker is not None),
            "court_calibration_diagnostic_only": bool(camera_tracker is not None),
        }
        if enable_jersey and visual is not None:
            checkpoint = dict(getattr(visual, "checkpoint_diagnostics", {}) or {})
            extra["jersey_backend"] = {
                "backend": checkpoint.get("backend", "unknown"),
                "architecture": checkpoint.get("architecture"),
                "model_name": checkpoint.get("model_name"),
                "checkpoint": str(getattr(visual, "model_path", "")) or None,
                "preprocessing": checkpoint.get("preprocessing"),
                "temporal_aggregation": checkpoint.get("temporal_aggregation"),
                "roster_policy": checkpoint.get("roster_policy"),
                "provider": checkpoint.get("provider"),
                "experimental_challenger": checkpoint.get("experimental_challenger", False),
            }
        if camera_tracker is not None:
            extra["court_calibration"] = {
                "model_id": getattr(camera_tracker, "model_id", None),
                "keypoint_schema": getattr(camera_tracker, "keypoint_schema", None),
                "keypoint_map_size": len(getattr(camera_tracker, "keypoint_map", {}) or {}),
            }
        if hasattr(detector, "ball_candidate_backend_metadata"):
            extra["ball_detector_backend"] = detector.ball_candidate_backend_metadata()

        summary_path = diagnostics.finish(extra=extra, performance=performance)
        if enable_jersey and _env_bool("BASKETBALL_IDENTITY_DEBUG", True):
            _generate_identity_debug(summary_path, diagnostics.run_dir)
        print(f"[SUMMARY] {summary_path.resolve()}", flush=True)
        print("\n✅ Analysis completed successfully.", flush=True)

    except Exception as exc:
        if pipeline is not None:
            try:
                pipeline.flush()
            except Exception:
                pass
        performance = (
            pipeline.profiler.snapshot(
                video_fps=float(getattr(session, "fps", 30.0) or 30.0)
            )
            if pipeline
            else {}
        )
        summary_path = diagnostics.finish(
            extra={"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
            performance=performance,
        )
        print(f"[SUMMARY] {summary_path.resolve()}", flush=True)
        raise
    finally:
        if conn is not None:
            try:
                if pipeline is not None:
                    pipeline.flush()
                conn.commit()
                conn.close()
            except Exception:
                pass
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def parse_time_to_seconds(time_str: str) -> float:
    if not time_str:
        return 0.0
    if ":" not in time_str:
        return float(time_str)
    parts = [float(part) for part in time_str.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported time format: {time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CourtFlux canonical P1/P2 analysis pipeline")
    parser.add_argument("--video", type=str, required=True, help="Local video path or YouTube URL")
    parser.add_argument("--start", type=str, default="0", help="Start in seconds or MM:SS")
    parser.add_argument("--end", type=str, default="", help="End in seconds or MM:SS")
    parser.add_argument("--model", type=str, default=settings.model, help="YOLO checkpoint path")
    parser.add_argument("--conf", type=float, default=0.40, help="Global YOLO detector confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default=settings.device, choices=["mps", "cpu", "cuda"])
    parser.add_argument("--tracker", type=str, default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--match-config", type=str, default=str(DEFAULT_MATCH_CONFIG_PATH))
    parser.add_argument("--reset-db", action="store_true")
    parser.add_argument("--no-jersey", action="store_true")
    parser.add_argument("--jersey-interval", type=int, default=24)
    parser.add_argument("--db-batch-size", type=int, default=100)
    parser.add_argument("--diagnostics-batch-size", type=int, default=100)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    main(
        video_source=args.video,
        start_time=parse_time_to_seconds(args.start),
        end_time=parse_time_to_seconds(args.end) if args.end else None,
        model_path=args.model,
        detector_conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        reset_db=args.reset_db,
        match_config_path=args.match_config,
        enable_jersey=not args.no_jersey,
        jersey_interval_frames=args.jersey_interval,
        db_batch_size=args.db_batch_size,
        diagnostics_batch_size=args.diagnostics_batch_size,
        persist_results=not args.no_persist,
    )
