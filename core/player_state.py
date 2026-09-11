from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlayerState:
    """Canonical P1 state for one tracked basketball player in one frame.

    This object is perception/state only. It deliberately does not encode possession,
    pressure, shot/pass/rebound truth, tactical intent, or resolved team/jersey identity.
    Referees are not represented as PlayerState; they remain separate detector evidence.
    """

    frame_id: int
    timestamp_sec: float | None
    track_id: int
    bbox: tuple[int, int, int, int]
    center_px: tuple[float, float]
    confidence: float = 0.0
    uncertainty: float = 1.0
    velocity_px_per_frame: tuple[float, float] = (0.0, 0.0)
    age_since_observation: int = 0
    source: str = "tracked_player_detection"
    court_position_m: tuple[float, float] | None = None
    speed_mps: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_id = int(self.frame_id)
        self.track_id = int(self.track_id)
        if self.timestamp_sec is not None:
            self.timestamp_sec = float(self.timestamp_sec)
        self.bbox = tuple(int(round(float(value))) for value in self.bbox[:4])  # type: ignore[assignment]
        if len(self.bbox) != 4:
            raise ValueError("PlayerState bbox must contain four coordinates")
        self.center_px = (float(self.center_px[0]), float(self.center_px[1]))
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        self.uncertainty = max(0.0, min(float(self.uncertainty), 1.0))
        self.velocity_px_per_frame = (
            float(self.velocity_px_per_frame[0]),
            float(self.velocity_px_per_frame[1]),
        )
        self.age_since_observation = max(0, int(self.age_since_observation))
        if self.court_position_m is not None:
            self.court_position_m = (
                float(self.court_position_m[0]),
                float(self.court_position_m[1]),
            )
        if self.speed_mps is not None:
            self.speed_mps = max(0.0, float(self.speed_mps))

    @property
    def visible(self) -> bool:
        """Current v1 states are emitted only for upstream-observed active tracks."""
        return self.age_since_observation == 0

    def position(self) -> tuple[float, float]:
        """Backward-compatible position helper; prefer court_position_m when available."""
        if self.court_position_m is not None:
            return self.court_position_m
        return self.center_px

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "center_px": list(self.center_px),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "velocity_px_per_frame": list(self.velocity_px_per_frame),
            "age_since_observation": self.age_since_observation,
            "source": self.source,
            "court_position_m": (
                list(self.court_position_m) if self.court_position_m is not None else None
            ),
            "speed_mps": self.speed_mps,
            "provenance": dict(self.provenance),
        }
