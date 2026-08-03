# ==========================================
# 1. Standard Library
# ==========================================
from pathlib import Path
import sqlite3
import sys

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
from services.vision.camera_tracker import CameraTracker
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
from services.session_config import SessionConfigurator
from services.video_session import VideoSession

def main():

  # --------------------------------------------------
  # Database
  # --------------------------------------------------
  # Set DB Path relative to project root
  PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
  DB_PATH = PROJECT_ROOT / "data" / "tracking.db"
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)

  conn = sqlite3.connect(DB_PATH)
  cursor = conn.cursor()
  db_service = DatabaseService(conn)

  # --------------------------------------------------
  # Ask user for session configuration
  # --------------------------------------------------

  session_cfg = SessionConfigurator().prompt()

  # --------------------------------------------------
  # Download video
  # --------------------------------------------------

  downloader = YouTubeDownloader(output_dir="data")

  video_path = downloader.get_local_video(session_cfg.youtube_url)

  # --------------------------------------------------
  # Open video session
  # --------------------------------------------------

  session = VideoSession(
      video_path=video_path,
      start_time=session_cfg.start_time,
      end_time=session_cfg.end_time,
  )

  session.open()

  success, first_frame = session.cap.read()

  if not success:
    raise RuntimeError(
        f"Unable to read frame at {session_cfg.start_time:.1f}s"
    )

  # --------------------------------------------------
  # Court calibration
  # --------------------------------------------------
  print("Calibrating Court...")
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

  print("Initializing EasyOCR...")

  reader = easyocr.Reader(
      ["en"],
      gpu=False,
  )

  visual = VisualAnalyzer(reader,profiling_path="data/match_config.json")

  # --------------------------------------------------
  # YOLO
  # --------------------------------------------------

  model = YOLO(settings.model)

  # --------------------------------------------------
  # Pipeline
  # --------------------------------------------------

  pipeline = Pipeline(
      gameplay_filter=GameplayFilter(
          min_court_ratio=0.15,      # Lower if alternate courts have lighter/darker wood
          min_motion_ratio=0.003,    # Increase if minor camera shake triggers false live action
          max_grace_frames=15,       # Frames to hold state through brief occlusions/cuts
          processing_scale=0.5     # Runs math on 1/2th resolution for peak performance
      ),
      detector = Detector(
        model=model,
        visual_analyzer=visual,
        profiling_path="data/match_config.json"  # <-- ADD THIS (update with your actual JSON filename)
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
  # Frame loader
  # --------------------------------------------------

  loader = FrameLoader(session)

  print(
      f"\n🚀 Processing video"
      f" ({session_cfg.start_time:.1f}s → {session_cfg.end_time:.1f}s)"
  )

  frame_id = 0

  while True:
    frame_state = loader.next()

    if frame_state is None:
      break

    if frame_state.timestamp > session_cfg.end_time:
      break

    # Attach required attributes for rendering and snapshots
    frame_state.frame_id = frame_id
    frame_state.court_polygon = court_polygon

    # Run pipeline processing & rendering
    frame_state = pipeline.process(frame_state)

    frame = (
        frame_state.annotated_frame
        if frame_state.annotated_frame is not None
        else frame_state.frame
    )

    cv2.imshow("Basketball AI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
      print("🛑 Analysis aborted by user.")
      break
          
    frame_id += 1

  # --------------------------------------------------
  # Shutdown
  # --------------------------------------------------
  pipeline.engine.metrics.export()
  
  conn.commit()
  conn.close()

  session.close()

  cv2.destroyAllWindows()

  print("\n✅ Analysis completed successfully.")


if __name__ == "__main__":
  main()