from pathlib import Path
import json
import sqlite3
import sys
import torch
import os
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
from ultralytics import YOLO

from services.vision.camera_tracker import CameraTracker
from core.config import settings
from services.profiler import PipelineProfiler
from services.database_service import DatabaseService
from services.persistence import db_setup
from services.ingestion.youtube_downloader import YouTubeDownloader
from services.vision.detector import Detector
from services.vision.identity_v2_detector import PerformanceDetector
from services.vision.visual_analyzer import VisualAnalyzer
from services.vision.team_only_analyzer import TeamOnlyAnalyzer
from services.vision.p1_jersey_analyzer import PerformanceJerseyAnalyzer
from services.analytics.engine import Engine
from services.frame_loader import FrameLoader
from services.gameplay_filter import GameplayFilter
from services.pipeline import Pipeline
from services.video_session import VideoSession
from services.diagnostics import RunDiagnostics

DEFAULT_MATCH_CONFIG_PATH = PROJECT_ROOT / "data" / "match_config.json"


def _env_bool(name: str, fallback: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return bool(fallback)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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
    detection_only: bool = False,
    enable_jersey: bool = True,
    jersey_interval_frames: int = 12,
    db_batch_size: int = 100,
    diagnostics_batch_size: int = 100,
    persist_results: bool = True,
):
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
        detection_only=detection_only,
        enable_jersey=enable_jersey,
        jersey_interval_frames=jersey_interval_frames,
        batch_size=diagnostics_batch_size,
    )
    diagnostics.capture_match_config(chosen_match_config)
    print(f"[RUN_DIR] {diagnostics.run_dir.resolve()}", flush=True)

    conn = None
    session = None
    pipeline = None

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
            downloader = YouTubeDownloader(output_dir="data")
            video_path = downloader.get_local_video(video_source)
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
        if detection_only:
            engine = None
            if court_calibration_enabled:
                print("📐 Detection-only mode: diagnostic court calibration enabled.", flush=True)
                camera_tracker = CameraTracker(first_frame=first_frame)
                court_polygon = camera_tracker.base_polygon
            else:
                print("⚡ Detection-only mode: camera/homography and analytics engine disabled.", flush=True)
                camera_tracker = None
                court_polygon = None
        else:
            print("📐 Calibrating court...", flush=True)
            camera_tracker = CameraTracker(first_frame=first_frame)
            court_polygon = camera_tracker.base_polygon
            homography = camera_tracker.initial_H
            camera_tracker = CameraTracker(first_frame, homography, court_polygon)
            engine = Engine(fps=int(round(session.fps or 30.0)))

        if detection_only:
            if enable_jersey:
                print("⚡ Jersey recognition enabled with lean local-only benchmark model.", flush=True)
                visual = PerformanceJerseyAnalyzer(
                    profiling_path=str(chosen_match_config),
                    device=resolved_device,
                    model_path=str(PROJECT_ROOT / "weights" / "jersey_number_recon.pt"),
                    sequence_length=8,
                )
            else:
                print("⚡ Jersey model disabled; using lightweight team-only classifier.", flush=True)
                visual = TeamOnlyAnalyzer(profiling_path=str(chosen_match_config))
        else:
            visual = VisualAnalyzer(reader=None, profiling_path=str(chosen_match_config))

        chosen_model_path = Path(chosen_model)
        if not chosen_model_path.is_absolute():
            chosen_model_path = PROJECT_ROOT / chosen_model_path
        if not chosen_model_path.exists():
            raise FileNotFoundError(f"Model not found: {chosen_model}. Select a checkpoint available on the worker.")

        model = YOLO(str(chosen_model_path))
        model.to(resolved_device)
        print(
            f"🧠 YOLO model={chosen_model} device={resolved_device} conf={detector_conf:.2f} "
            f"imgsz={imgsz} tracker={tracker}", flush=True,
        )

        if detection_only:
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
        else:
            detector = Detector(
                model=model,
                visual_analyzer=visual,
                profiling_path=str(chosen_match_config),
                device=resolved_device,
                detector_conf=detector_conf,
                imgsz=imgsz,
                iou=0.5,
                tracker=tracker,
                max_det=40,
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
            engine=engine,
            database=db_service,
            renderer=None,
            profiler=profiler,
            db_conn=conn,
            detection_only=detection_only,
            persist_results=persist_results,
        )

        loader = FrameLoader(session)
        print(
            f"\n🚀 Processing video ({start_time:.1f}s → "
            f"{end_time_val if end_time_val != float('inf') else 'End'}s)", flush=True,
        )

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
                processed_sec = max(0.0, min(current_timestamp, float(end_time_val)) - float(start_time))
                percent = min(100.0, max(0.0, (processed_sec / total_window_sec) * 100.0))
                print(
                    "[PROGRESS] " + json.dumps(
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

        if total_window_sec:
            print(
                "[PROGRESS] " + json.dumps(
                    {
                        "timestamp_sec": round(float(end_time_val), 3),
                        "processed_seconds": round(total_window_sec, 3),
                        "total_seconds": round(total_window_sec, 3),
                        "percent": 100.0,
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
            "mode": "detection_only" if detection_only else "full",
            "persist_results": persist_results,
            "court_calibration_enabled": bool(camera_tracker is not None),
            "court_calibration_diagnostic_only": bool(detection_only and camera_tracker is not None),
        }
        if camera_tracker is not None:
            extra["court_calibration"] = {
                "model_id": getattr(camera_tracker, "model_id", None),
                "keypoint_schema": getattr(camera_tracker, "keypoint_schema", None),
                "keypoint_map_size": len(getattr(camera_tracker, "keypoint_map", {}) or {}),
            }
        if detection_only and hasattr(detector, "ball_candidate_backend_metadata"):
            extra["ball_detector_backend"] = detector.ball_candidate_backend_metadata()
        if engine is not None:
            metrics_path = engine.metrics.export()
            extra["legacy_metrics_file"] = str(metrics_path)

        summary_path = diagnostics.finish(extra=extra, performance=performance)
        print(f"[SUMMARY] {summary_path.resolve()}", flush=True)
        print("\n✅ Analysis completed successfully.", flush=True)

    except Exception as exc:
        if pipeline is not None:
            try:
                pipeline.flush()
            except Exception:
                pass
        performance = pipeline.profiler.snapshot(video_fps=float(getattr(session, "fps", 30.0) or 30.0)) if pipeline else {}
        summary_path = diagnostics.finish(
            extra={"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
            performance=performance,
        )
        print(f"[SUMMARY] {summary_path.resolve()}", flush=True)
        print(f"❌ Analysis failed: {type(exc).__name__}: {exc}", flush=True)
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
    parts = [float(p) for p in time_str.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unsupported time format: {time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball Analytics Beta Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Local video path or YouTube URL")
    parser.add_argument("--start", type=str, default="0", help="Start in seconds or MM:SS")
    parser.add_argument("--end", type=str, default="", help="End in seconds or MM:SS")
    parser.add_argument("--model", type=str, default=settings.model, help="YOLO checkpoint path")
    parser.add_argument("--conf", type=float, default=0.40, help="Global YOLO detector confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--device", type=str, default=settings.device, choices=["mps", "cpu", "cuda"])
    parser.add_argument("--tracker", type=str, default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--match-config", type=str, default=str(DEFAULT_MATCH_CONFIG_PATH), help="Match config JSON path")
    parser.add_argument("--reset-db", action="store_true", help="Reset SQLite before the run")
    parser.add_argument("--detection-only", action="store_true", help="Skip tactical analytics; court calibration can be enabled diagnostically via env")
    parser.add_argument("--no-jersey", action="store_true", help="Disable temporal jersey recognition")
    parser.add_argument("--jersey-interval", type=int, default=12, help="Frames between jersey attempts per tracked player")
    parser.add_argument("--db-batch-size", type=int, default=100, help="Frames buffered per SQLite commit")
    parser.add_argument("--diagnostics-batch-size", type=int, default=100, help="Frame records buffered per JSONL write")
    parser.add_argument("--no-persist", action="store_true", help="Do not write SQLite during the benchmark")

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
        detection_only=args.detection_only,
        enable_jersey=not args.no_jersey,
        jersey_interval_frames=args.jersey_interval,
        db_batch_size=args.db_batch_size,
        diagnostics_batch_size=args.diagnostics_batch_size,
        persist_results=not args.no_persist,
    )
