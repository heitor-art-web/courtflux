from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from typing import Any


P3_SOURCE_EVENT_CONTRACT_VERSION = "p3.source_event.v1"
P3_DERIVATION_CONTRACT_VERSION = "p3.derivation.v1"


class DerivationState(str, Enum):
    """Lifecycle state for reprocessable P3 derived data."""

    CURRENT = "current"
    STALE = "stale"
    SUPERSEDED = "superseded"
    FAILED = "failed"


@dataclass(slots=True)
class P3SourceEvent:
    """Stable, loss-minimizing boundary between P2 basketball events and P3.

    P3 consumes this shape instead of importing P2 engine internals. Track identity is
    preserved even when the still-evolving P2 game-player resolver cannot yet provide a
    game_player_id. Candidate/confirmed state and uncertainty are never promoted here.
    """

    source_event_id: str
    game_id: str
    analysis_run_id: str
    event_type: str
    decision: str
    confidence: float
    uncertainty: float
    frame_start: int
    frame_end: int
    start_ms: int | None
    end_ms: int | None
    actor_track_id: int | None = None
    secondary_track_id: int | None = None
    actor_game_player_id: str | None = None
    secondary_game_player_id: str | None = None
    team_id: str | None = None
    location_court_m: tuple[float, float] | None = None
    evidence: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    source_contract_version: str = "core.basketball_event.v1"
    source_runtime_version: str | None = None
    contract_version: str = P3_SOURCE_EVENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        self.source_event_id = str(self.source_event_id)
        self.game_id = str(self.game_id)
        self.analysis_run_id = str(self.analysis_run_id)
        self.event_type = str(self.event_type)
        self.decision = str(self.decision)
        self.confidence = max(0.0, min(float(self.confidence), 1.0))
        self.uncertainty = max(0.0, min(float(self.uncertainty), 1.0))
        self.frame_start = max(0, int(self.frame_start))
        self.frame_end = max(self.frame_start, int(self.frame_end))
        if self.start_ms is not None:
            self.start_ms = max(0, int(self.start_ms))
        if self.end_ms is not None:
            self.end_ms = max(self.start_ms or 0, int(self.end_ms))
        if self.actor_track_id is not None:
            self.actor_track_id = int(self.actor_track_id)
        if self.secondary_track_id is not None:
            self.secondary_track_id = int(self.secondary_track_id)
        if self.actor_game_player_id is not None:
            self.actor_game_player_id = str(self.actor_game_player_id)
        if self.secondary_game_player_id is not None:
            self.secondary_game_player_id = str(self.secondary_game_player_id)
        if self.team_id is not None:
            self.team_id = str(self.team_id)
        if self.location_court_m is not None:
            self.location_court_m = (
                float(self.location_court_m[0]),
                float(self.location_court_m[1]),
            )
        self.evidence = tuple(str(item) for item in self.evidence if str(item))
        self.attributes = dict(self.attributes)
        self.provenance = dict(self.provenance)
        self.source_contract_version = str(self.source_contract_version)
        if self.source_runtime_version is not None:
            self.source_runtime_version = str(self.source_runtime_version)
        self.contract_version = str(self.contract_version)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_event_id": self.source_event_id,
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "event_type": self.event_type,
            "decision": self.decision,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "actor_track_id": self.actor_track_id,
            "secondary_track_id": self.secondary_track_id,
            "actor_game_player_id": self.actor_game_player_id,
            "secondary_game_player_id": self.secondary_game_player_id,
            "team_id": self.team_id,
            "location_court_m": (
                list(self.location_court_m) if self.location_court_m is not None else None
            ),
            "evidence": list(self.evidence),
            "attributes": dict(self.attributes),
            "provenance": dict(self.provenance),
            "source_contract_version": self.source_contract_version,
            "source_runtime_version": self.source_runtime_version,
            "contract_version": self.contract_version,
        }


@dataclass(slots=True)
class P3DerivationSpec:
    """Version/config identity for a reprocessable P3 derived dataset.

    The fingerprint deliberately excludes timestamps and mutable lifecycle state. The
    same source versions + contracts + configuration must produce the same identity.
    """

    game_id: str
    analysis_run_id: str
    derivation_key: str
    derivation_version: str
    configuration_version: str
    source_contract_version: str = P3_SOURCE_EVENT_CONTRACT_VERSION
    contract_version: str = P3_DERIVATION_CONTRACT_VERSION
    source_versions: dict[str, str] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.game_id = str(self.game_id)
        self.analysis_run_id = str(self.analysis_run_id)
        self.derivation_key = str(self.derivation_key)
        self.derivation_version = str(self.derivation_version)
        self.configuration_version = str(self.configuration_version)
        self.source_contract_version = str(self.source_contract_version)
        self.contract_version = str(self.contract_version)
        self.source_versions = {str(k): str(v) for k, v in self.source_versions.items()}
        self.configuration = dict(self.configuration)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "derivation_key": self.derivation_key,
            "derivation_version": self.derivation_version,
            "configuration_version": self.configuration_version,
            "source_contract_version": self.source_contract_version,
            "contract_version": self.contract_version,
            "source_versions": dict(sorted(self.source_versions.items())),
            "configuration": self.configuration,
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.fingerprint_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self.fingerprint_payload(), "fingerprint": self.fingerprint()}


def needs_reprocessing(previous_fingerprint: str | None, spec: P3DerivationSpec) -> bool:
    """Return True when a missing/different derivation fingerprint must be rebuilt."""

    return not previous_fingerprint or str(previous_fingerprint) != spec.fingerprint()
