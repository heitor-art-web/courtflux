from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(slots=True)
class FrameState:
    """
    Unified object passed through the entire video processing pipeline.

    Single source of truth using standard Python dictionaries for player/ball detections,
    supporting camera tracking metadata, backboard ROI, and backward-compatible aliases.
    """

    # Primary Video Metadata
    frame_id: int
    timestamp: float
    frame: np.ndarray
    fps: float = 30.0

    # Processed Frame Visuals
    annotated_frame: Optional[np.ndarray] = None

    # Camera Tracking & Geometry
    homography_matrix: Optional[np.ndarray] = None
    court_polygon: Optional[np.ndarray] = None
    camera_polygon: Optional[np.ndarray] = None
    backboard_roi: Optional[Tuple[int, int, int, int]] = None

    # Detected Entities (Pure Dictionaries)
    players: List[Dict[str, Any]] = field(default_factory=list)
    ball: Optional[Dict[str, Any]] = None
    ball_handler: Optional[Dict[str, Any]] = None

    # Analytics & Gameplay State
    offensive_team: Optional[str] = None
    is_valid_gameplay: bool = True
    is_snapshot_moment: bool = False

    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Property Aliases (Ensures backward compatibility across models)
    # ------------------------------------------------------------------

    @property
    def homography(self) -> Optional[np.ndarray]:
        """Alias for homography_matrix."""
        return self.homography_matrix

    @homography.setter
    def homography(self, value: Optional[np.ndarray]):
        self.homography_matrix = value

    @property
    def camera(self) -> Optional[np.ndarray]:
        """Alias for camera_polygon."""
        return self.camera_polygon

    @camera.setter
    def camera(self, value: Optional[np.ndarray]):
        self.camera_polygon = value

    @property
    def valid_gameplay(self) -> bool:
        """Alias for is_valid_gameplay."""
        return self.is_valid_gameplay

    @valid_gameplay.setter
    def valid_gameplay(self, value: bool):
        self.is_valid_gameplay = value

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def add_event(self, event: str) -> None:
        self.events.append(event)

    def add_metric(self, name: str, value: Any) -> None:
        self.metrics[name] = value