from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BasketballEventType(str, Enum):
    """Canonical P2 event primitives.

    These are basketball-understanding outputs, not P1 perception state and not P3
    statistics. Candidate event types deliberately remain explicit so downstream code
    can aggregate only decisions whose confidence/provenance satisfy its own contract.
    """

    POSSESSION_ACQUIRED = "possession_acquired"
    POSSESSION_CHANGE = "possession_change"
    PASS_CANDIDATE = "pass_candidate"
    TURNOVER_CANDIDATE = "turnover_candidate"
    LOOSE_BALL_STARTED = "loose_ball_started"
    LOOSE_BALL_RECOVERED = "loose_ball_recovered"
    SHOT_ATTEMPT = "shot_attempt"
    REBOUND_CANDIDATE = "rebound_candidate"


class BallControlState(str, Enum):
    """P2 temporal interpretation of who, if anyone, controls the ball.

    This is intentionally downstream of P1 BallState. It may be UNKNOWN even while a
    ball is visible, and it never forces identity beyond stable track/team evidence.
    """

    UNKNOWN = "unknown"
    POSSESSED = "possessed"
    LOOSE = "loose"
    PASS_FLIGHT = "pass_flight"
    SHOT_FLIGHT = "shot_flight"


class EventDecision(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"


class ShotOutcome(str, Enum):
    UNKNOWN = "unknown"
    MADE = "made"
    MISSED = "missed"


@dataclass(slots=True)
class BasketballEvent:
    """Timestamped, confidence-aware P2 basketball event primitive."""

    event_id: str
    event_type: BasketballEventType
    frame_start: int
    frame_end: int
    timestamp_start_sec: float | None
    timestamp_end_sec: float | None
    confidence: float
    uncertainty: float
    decision: EventDecision = EventDecision.CANDIDATE
    actor_track_id: int | None = None
    secondary_track_id: int | None = None
    team_id: str | None = None
    location_court_m: tuple[float, float] | None = None
    evidence: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_id = str(self.event_id)
        self.frame_start = int(self.frame_start)
        self.frame_end = max(self.frame_start, int(self.frame_end))
        if self.timestamp_start_sec is not None:
            self.timestamp_start_sec = float(self.timestamp_start_sec)
        if self.timestamp_end_sec is not None:
            self.timestamp_end_sec = float(self.timestamp_end_sec)
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        self.uncertainty = max(0.0, min(float(self.uncertainty), 1.0))
        if self.actor_track_id is not None:
            self.actor_track_id = int(self.actor_track_id)
        if self.secondary_track_id is not None:
            self.secondary_track_id = int(self.secondary_track_id)
        if self.team_id is not None:
            self.team_id = str(self.team_id)
        if self.location_court_m is not None:
            self.location_court_m = (
                float(self.location_court_m[0]),
                float(self.location_court_m[1]),
            )
        self.evidence = tuple(str(item) for item in self.evidence if str(item))

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "timestamp_start_sec": self.timestamp_start_sec,
            "timestamp_end_sec": self.timestamp_end_sec,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "decision": self.decision.value,
            "actor_track_id": self.actor_track_id,
            "secondary_track_id": self.secondary_track_id,
            "team_id": self.team_id,
            "location_court_m": (
                list(self.location_court_m) if self.location_court_m is not None else None
            ),
            "evidence": list(self.evidence),
            "provenance": dict(self.provenance),
            "attributes": dict(self.attributes),
        }
