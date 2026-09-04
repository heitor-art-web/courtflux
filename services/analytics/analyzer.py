from pathlib import Path
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
from services.vision.visual_analyzer import VisualAnalyzer
from services.analytics.engine import Engine
from services.frame_loader import FrameLoader
from services.gameplay_filter import GameplayFilter
from services.pipeline import Pipeline
from services.video_session import VideoSession
from services.diagnostics import RunDiagnostics

DEFAULT_MATCH_CONFIG_PATH = PROJECT_ROOT / "data" / "match_config.json"


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
    )
    diagnostics.capture_match_config(chosen_match_config)
    print(f"[RUN_DIR] {diagnostics.run_dir.resolve()}", flush=True)

    conn = None
    session = None
    pipeline = None

    try:
        if not chosen_match_config.exists():
            raise FileNotFoundError(f"Match config not found: {chosen_match_config}")

        # --------------------------------------------------
        # Database
        # --------------------------------------------------
        db_path = PROJECT_ROOT / "data" / "tracking.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if reset_db or not db_path.exists():
            db_setup.initialize_database()

        conn = sqlite3.connect(db_path)
        db_service = DatabaseService(conn)

        # --------------------------------------------------
        # Download / Resolve Video
        # --------------------------------------------------
        print(f"🎬 Initializing video source: {video_source}", flush=True)

        if "youtube.com" in video_source or "youtu.be" in video_source:
            downloader = YouTubeDownloader(output_dir="data")
            video_path = downloader.get_local_video(video_source)
        else:
            video_path = video_source

        # --------------------------------------------------
        # Open video session
        # --------------------------------------------------
        end_time_val = end_time if end_time is not None else float("inf")
        session = VideoSession(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time_val,
        )
        session.open()

        success, first_frame = session.cap.read()
        if not success:
            raise RuntimeError(f"Unable to read frame at {start_time:.1f}s")

        session.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        # --------------------------------------------------
        # Court calibration (data only; no 2D court rendering)
        # --------------------------------------------------
        print("📐 Calibrating court...", flush=True)
        camera_tracker = CameraTracker(first_frame=first_frame)
        court_polygon = camera_tracker.base_polygon
        homography = camera_tracker.initial_H
        camera_tracker = CameraTracker(first_frame, homography, court_polygon)

        # --------------------------------------------------
        # Visual metadata analysis
        # --------------------------------------------------
        visual = VisualAnalyzer(reader=None, profiling_path=str(chosen_match_config))

        # --------------------------------------------------
        # YOLO
        # --------------------------------------------------
        chosen_model_path = Path(chosen_model)
        if not chosen_model_path.is_absolute():
            chosen_model_path = PROJECT_ROOT / chosen_model_path
        if not chosen_model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {chosen_model}. Select a checkpoint available on the worker."
            )

        model = YOLO(str(chosen_model_path))
        model.to(resolved_device)
        print(
            f"🧠 YOLO model={chosen_model} device={resolved_device} "
            f"conf={detector_conf:.2f} imgsz={imgsz} tracker={tracker}",
            flush=True,
        )

        # --------------------------------------------------
        # Pipeline
        # --------------------------------------------------
        pipeline = Pipeline(
            gameplay_filter=GameplayFilter(
                min_court_ratio=0.15,
                min_motion_ratio=0.003,
                max_grace_frames=15,
                processing_scale=0.5,
            ),
            detector=Detector(
                model=model,
                visual_analyzer=visual,
                profiling_path=str(chosen_match_config),
                device=resolved_device,
                detector_conf=detector_conf,
                imgsz=imgsz,
                iou=0.5,
                tracker=tracker,
                max_det=40,
            ),
            camera_tracker=camera_tracker,
            backboard_roi=None,
            engine=Engine(fps=int(round(session.fps or 30.0))),
            database=db_service,
            renderer=None,
            profiler=PipelineProfiler(),
            db_conn=conn,
        )

        loader = FrameLoader(session)

        print(
            f"\n🚀 Processing video ({start_time:.1f}s → "
            f"{end_time_val if end_time_val != float('inf') else 'End'}s)",
            flush=True,
        )

        frame_id = 0
        while True:
            frame_state = loader.next()
            if frame_state is None:
                break
            if frame_state.timestamp > end_time_val:
                break

            frame_state.frame_id = frame_id
            frame_state.court_polygon = court_polygon
            frame_state = pipeline.process(frame_state)
            diagnostics.log_frame(frame_state)
            frame_id += 1

        metrics_path = pipeline.engine.metrics.export()
        summary_path = diagnostics.finish(
            extra={
                "status": "completed",
                "video_path": str(video_path),
                "legacy_metrics_file": str(metrics_path),
            }
        )
        print(f"[SUMMARY] {summary_path.resolve()}", flush=True)
        print("\n✅ Analysis completed successfully.", flush=True)

    except Exception as exc:
        summary_path = diagnostics.finish(
            extra={
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"[SUMMARY] {summary_path.resolve()}", flush=True)
        print(f"❌ Analysis failed: {type(exc).__name__}: {exc}", flush=True)
        raise

    finally:
        if conn is not None:
            try:
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
    parser.add_argument("--reset-db", action="store_true", help="Reset the beta SQLite database before the run")

    args = parser.parse_args()
    start_sec = parse_time_to_seconds(args.start)
    end_sec = parse_time_to_seconds(args.end) if args.end else None

    main(
        video_source=args.video,
        start_time=start_sec,
        end_time=end_sec,
        model_path=args.model,
        detector_conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        reset_db=args.reset_db,
        match_config_path=args.match_config,
    )
