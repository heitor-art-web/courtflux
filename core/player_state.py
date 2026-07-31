from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PlayerState:
    """
    Represents one tracked player.
    """
    track_id: int
    team: str
    shirt_number: str
    x_m: float
    y_m: float
    speed_mps: float
    bbox: tuple[int, int, int, int]
    has_ball: bool = False
    pressure: Optional[float] = None
    confidence: float = 0.0
    visible: bool = True
    
    def position(self) -> tuple[float, float]:
        return self.x_m, self.y_m