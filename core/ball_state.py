from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BallStateStatus(str, Enum):
    """P1 ball availability semantics.

    OBSERVED: direct detector evidence selected by the ball pipeline.
    PREDICTED: short-gap temporal prediction; never treated as a direct observation.
    MISSING: no approved ball state for the frame.
    """

    OBSERVED = "observed"
    PREDICTED = "predicted"
    MISSING = "missing"


@dataclass(slots=True)
class BallState:
    """Canonical P1 ball state for one frame.

    This object is state/evidence only. It must not encode possession, shot, pass,
    rebound, made/miss or any other P2 basketball event truth.
    """

    frame_id: int
    timestamp_sec: float | None
    status: BallStateStatus
    x_px: float | None = None
    y_px: float | None = None
    box: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    uncertainty: float = 1.0
    velocity_px_per_frame: tuple[float, float] = (0.0, 0.0)
    age_since_detection: int = 0
    source: str = "missing"
    trajectory_score: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_id = int(self.frame_id)
        if self.timestamp_sec is not None:
            self.timestamp_sec = float(self.timestamp_sec)
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        self.uncertainty = max(0.0, min(float(self.uncertainty), 1.0))
        self.age_since_detection = max(0, int(self.age_since_detection))
        self.velocity_px_per_frame = (
            float(self.velocity_px_per_frame[0]),
            float(self.velocity_px_per_frame[1]),
        )
        if self.x_px is not None:
            self.x_px = float(self.x_px)
        if self.y_px is not None:
            self.y_px = float(self.y_px)
        if self.trajectory_score is not None:
            self.trajectory_score = max(0.0, min(float(self.trajectory_score), 1.0))

        if self.status is BallStateStatus.MISSING:
            # Missing is explicit UNKNOWN: no synthetic position is exposed upstream.
            self.x_px = None
            self.y_px = None
            self.box = None
            self.confidence = 0.0
            self.uncertainty = 1.0
            self.trajectory_score = None
        elif self.x_px is None or self.y_px is None:
            raise ValueError("observed/predicted BallState requires a position")

    @property
    def visible(self) -> bool:
        """Backward-compatible semantic: True only for direct observation."""
        return self.status is BallStateStatus.OBSERVED

    @property
    def available(self) -> bool:
        """True when a position exists, whether observed or short-gap predicted."""
        return self.status is not BallStateStatus.MISSING

    @property
    def predicted(self) -> bool:
        return self.status is BallStateStatus.PREDICTED

    def position(self) -> tuple[float, float] | None:
        if self.x_px is None or self.y_px is None:
            return None
        return self.x_px, self.y_px

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "status": self.status.value,
            "x_px": self.x_px,
            "y_px": self.y_px,
            "box": list(self.box) if self.box is not None else None,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "velocity_px_per_frame": list(self.velocity_px_per_frame),
            "age_since_detection": self.age_since_detection,
            "source": self.source,
            "trajectory_score": self.trajectory_score,
            "provenance": dict(self.provenance),
        }
