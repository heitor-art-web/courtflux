from dataclasses import dataclass


@dataclass(slots=True)
class BallState:
    """
    Ball information.
    """
    x_px: int
    y_px: int
    visible: bool = True
    confidence: float = 0.0

    def position(self):

        return self.x_px, self.y_px