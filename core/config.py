"""
Global configuration for the Basketball AI pipeline.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    # -----------------------------
    # YOLO & Hardware Optimization
    # -----------------------------
    model: str = "weights/Model_v0.1.pt"  #Switch to the most up to date model
    tracker: str = "botsort.yaml"
    confidence: float = 0.15
    device: str = "mps"             # Use "mps" for Apple Silicon (or "cuda" for NVIDIA GPUs)
    half: bool = True               # Enable FP16 half-precision for faster inference
    imgsz: int = 640                # Standard inference resolution

    # -----------------------------
    # Court
    # -----------------------------
    min_court_visibility: float = 0.12
    max_court_loss_frames: int = 15

    # -----------------------------
    # OCR (Performance Bottleneck)
    # -----------------------------
    ocr_interval: int = 140          # Max Limit is 150 to 300

    # -----------------------------
    # Database
    # -----------------------------
    database_path: str = "data/tracking.db"
    commit_interval: int = 50

    # -----------------------------
    # Video
    # -----------------------------
    default_fps: float = 30.0

    # -----------------------------
    # Court dimensions (FIBA)
    # -----------------------------
    court_length: float = 28.0
    court_width: float = 15.0


settings = Settings()