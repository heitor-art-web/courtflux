from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict, Optional, Any

@dataclass
class FrameState:
    frame_id: int
    timestamp: float
    frame: np.ndarray
    players: List[Dict[str, Any]] = field(default_factory=list)
    ball: Optional[Dict[str, Any]] = None
    camera: Optional[np.ndarray] = field(default=None)
    homography: Optional[np.ndarray] = field(default=None)  
    events: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_valid_gameplay: bool = True
    annotated_frame: Optional[np.ndarray] = field(default=None)
    is_snapshot_moment: bool = False
    court_polygon: Optional[np.ndarray] = None