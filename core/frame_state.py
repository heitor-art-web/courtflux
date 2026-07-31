from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .player_state import PlayerState
from .ball_state import BallState


@dataclass(slots=True)
class FrameState:
    """
    Object passed through the whole pipeline.

    Every module reads and writes only this object.
    """

    frame_id: int

    timestamp: float

    frame: np.ndarray

    fps: float

    annotated_frame: np.ndarray | None = None

    homography_matrix: np.ndarray | None = None

    court_polygon: np.ndarray | None = None

    players: list[PlayerState] = field(default_factory=list)

    ball: BallState | None = None

    metrics: dict[str, Any] = field(default_factory=dict)

    events: list[str] = field(default_factory=list)

    offensive_team: str | None = None

    ball_handler: PlayerState | None = None

    valid_gameplay: bool = True
    
    annotated_frame = None
    
    is_snapshot_moment: bool = False
    
    # preenchidos pelo detector
    players: List[Dict[str, Any]] = field(default_factory=list)
    ball: Optional[Dict[str, Any]] = None

    # preenchidos pelo camera tracker
    camera_polygon: Optional[np.ndarray] = None
    homography_matrix: Optional[np.ndarray] = None

    # analytics
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

    # gameplay filter
    is_valid_gameplay: bool = True

    def add_event(self, event: str):

        self.events.append(event)

    def add_metric(self, name: str, value: Any):

        self.metrics[name] = value
        
        