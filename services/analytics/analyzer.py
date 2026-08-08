# ==========================================
# 1. Standard Library
# ==========================================
from pathlib import Path
import sqlite3
import sys
import torch
import os
import argparse

# Anchor project root and insert it into Python's module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# 2. Third-Party Libraries
# ==========================================
import cv2
import easyocr
from ultralytics import YOLO

# ==========================================
# 3. Local Application Modules
# ==========================================

# Core Config & Utilities
from services.vision.camera_tracker import CameraTracker
from core.config import settings
from services.profiler import PipelineProfiler

# Persistence & Database Services
from services.database_service import DatabaseService
from services.persistence import db_setup

# Ingestion Services
from services.ingestion.youtube_downloader import YouTubeDownloader

# Vision & Tracking Services
from services.vision.detector import Detector
from services.vision.visual_analyzer import VisualAnalyzer
from services.vision.radar_visualizer import RadarVisualizer

# Analytics Services
from services.analytics.engine import Engine

# Pipeline, Session & Rendering Services
from services.frame_loader import FrameLoader
from services.gameplay_filter import GameplayFilter
from services.pipeline import Pipeline
from services.renderer import Renderer
from services.session_config import SessionConfigurator # Retained but bypassed for UI
from services.video_session import VideoSession

def main(video_source: str, start_time: float, end_time: float):
    # --------------------------------------------------
    # Database
    # --------------------------------------------------
    DB_PATH = PROJECT_ROOT / "data" / "tracking.db"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    db_service = DatabaseService(conn)

    # --------------------------------------------------
    # Download / Resolve Video
    # --------------------------------------------------
    print(f"🎬 Initializing video source: {video_source}", flush=True)
    
    if "youtube.com" in video_source or "youtu.be" in video_source:
        downloader = YouTubeDownloader(output_dir="data")
        video_path = downloader.get_local_video(video_source)
    else:
        video_path = video_source  # Assume it's a local file path

    # --------------------------------------------------
    # Open video session
    # --------------------------------------------------
    end_time_val = end_time if end_time else float('inf')

    session = VideoSession(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time_val,
    )

    session.open()

    success, first_frame = session.cap.read()

    if not success:
        print(f"❌ Error: Unable to read frame at {start_time:.1f}s", flush=True)
        sys.exit(1)

    # --------------------------------------------------
    # Court calibration
    # --------------------------------------------------
    print("📐 Calibrating Court...", flush=True)
    camera_tracker = CameraTracker(first_frame=first_frame)

    court_polygon = camera_tracker.base_polygon
    homography = camera_tracker.initial_H

    camera_tracker = CameraTracker(
        first_frame,
        homography,
        court_polygon,
    )
    
    # --------------------------------------------------
    # OCR
    # --------------------------------------------------
    print("📝 Initializing EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True)
    visual = VisualAnalyzer(reader, profiling_path="data/match_config.json")

    # --------------------------------------------------
    # YOLO
    # --------------------------------------------------
    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = YOLO(settings.model)
    model.to(DEVICE)

    print(f"🧠 YOLO model : {settings.model} device set to: {DEVICE}", flush=True)

    # --------------------------------------------------
    # Pipeline
    # --------------------------------------------------
    pipeline = Pipeline(
        gameplay_filter=GameplayFilter(
            min_court_ratio=0.15,
            min_motion_ratio=0.003,
            max_grace_frames=15,
            processing_scale=0.5
        ),
        detector = Detector(
            model=model,
            visual_analyzer=visual,
            profiling_path="data/match_config.json"
        ),
        camera_tracker=camera_tracker,
        backboard_roi= None,
        engine=Engine(),
        database=db_service,
        renderer=Renderer(save_snapshots=True),
        profiler=PipelineProfiler(),
        db_conn=conn
    )

    # --------------------------------------------------
    # Frame loader & Preview Setup
    # --------------------------------------------------
    loader = FrameLoader(session)
    
    preview_dir = PROJECT_ROOT / "data" / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "latest_preview.jpg"

    print(f"\n🚀 Processing video ({start_time:.1f}s → {end_time_val if end_time_val != float('inf') else 'End'}s)", flush=True)

    frame_id = 0

    while True:
        frame_state = loader.next()

        if frame_state is None:
            break

        if frame_state.timestamp > end_time_val:
            break

        # Attach required attributes
        frame_state.frame_id = frame_id
        frame_state.court_polygon = court_polygon

        # Run pipeline processing & rendering
        frame_state = pipeline.process(frame_state)

        frame = (
            frame_state.annotated_frame
            if frame_state.annotated_frame is not None
            else frame_state.frame
        )

        # --------------------------------------------------
        # UI Communication (Instead of cv2.imshow)
        # --------------------------------------------------
        # Emit a preview frame to PySide every 5 frames
        if frame_id % 30 == 0:
            h, w, _ = frame.shape
            preview_w = 640
            preview_h = int(h * (preview_w / w))
            resized = cv2.resize(frame, (preview_w, preview_h))
            
            cv2.imwrite(str(preview_path), resized)
            print(f"[PREVIEW] {preview_path.resolve()}", flush=True)
            
        frame_id += 1

    # --------------------------------------------------
    # Shutdown
    # --------------------------------------------------
    pipeline.engine.metrics.export()
    
    conn.commit()
    conn.close()
    session.close()

    print("\n✅ Analysis completed successfully.", flush=True)


def parse_time_to_seconds(time_str: str) -> float:
    """Converts 'MM:SS' or raw second strings to floats."""
    if not time_str:
        return 0.0
    if ":" in time_str:
        parts = time_str.split(":")
        return float(parts[0]) * 60 + float(parts[1])
    return float(time_str)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball Analytics Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Local video file path or YouTube URL")
    parser.add_argument("--start", type=str, default="0", help="Start timestamp in seconds or MM:SS")
    parser.add_argument("--end", type=str, default="", help="End timestamp in seconds or MM:SS")
      
    args = parser.parse_args()

    start_sec = parse_time_to_seconds(args.start)
    end_sec = parse_time_to_seconds(args.end) if args.end else None

    # Forward the CLI arguments to our main function
    main(args.video, start_sec, end_sec)