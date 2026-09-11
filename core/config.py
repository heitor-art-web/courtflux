"""
Global configuration for the Basketball AI pipeline.

P1 detector policy is intentionally explicit here. Static detector calibration thresholds
and downstream state policies are different concerns; the ball candidate threshold below
is the operational value frozen after the v1.1.0 promotion/calibration work.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class Settings:
    # -----------------------------
    # Detector & Hardware
    # -----------------------------
    # Current P1 detector baseline. v1.0.0 remains an evaluation/regression reference.
    model: str = "weights/model_v1.1.0.pt"
    detector_version: str = "1.1.0"
    detector_baseline_status: str = "p1_current_baseline"
    tracker: str = "botsort.yaml"

    # Legacy/global confidence used by pipeline stages that have not yet adopted
    # class-specific thresholds. Do not use this as the BallState candidate threshold.
    confidence: float = 0.15
    device: str = "mps"             # Use "mps" for Apple Silicon (or "cuda" for NVIDIA GPUs)
    half: bool = True               # Enable FP16 half-precision for faster inference
    imgsz: int = 960                # Frozen P1 detector evaluation/integration resolution

    # -----------------------------
    # P1 Ball operational baseline
    # -----------------------------
    # model_v1.1.0 + trajectory configuration frozen after the promotion sweep.
    ball_candidate_confidence: float = 0.075
    ball_max_prediction_gap: int = 4
    ball_trajectory_max_distance_norm: float = 0.16
    ball_trajectory_max_speed_residual_norm: float = 0.12
    ball_trajectory_max_acceleration_norm: float = 0.14
    ball_trajectory_min_score: float = 0.65
    ball_trajectory_protect_direct_confidence: float | None = None
    ball_trajectory_override_score_margin: float = 0.0

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
