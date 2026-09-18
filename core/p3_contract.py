from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Iterable


P3_SOURCE_EVENT_CONTRACT_VERSION = "p3.source_event.v1"
P3_DERIVATION_CONTRACT_VERSION = "p3.derivation.v1"

P3_SUPPORTED_SOURCE_EVENT_TYPES = frozenset(
    {
        "possession_acquired",
        "possession_change",
        "pass_candidate",
        "turnover_candidate",
        "loose_ball_started",
        "loose_ball_recovered",
        "loose_ball_ended",
        "shot_attempt",
        "rebound_candidate",
    }
)
P3_SUPPORTED_SOURCE_DECISIONS = frozenset({"candidate", "confirmed"})

P3_SOURCE_EVENT_FIELDS = (
    "source_event_id",
    "game_id",
    "analysis_run_id",
    "event_type",
    "decision",
    "confidence",
    "uncertainty",
    "frame_start",
    "frame_end",
    "start_ms",
    "end_ms",
    "actor_track_id",
    "secondary_track_id",
    "actor_game_player_id",
    "secondary_game_player_id",
    "team_id",
    "location_court_m",
    "evidence",
    "attributes",
    "provenance",
    "source_contract_version",
    "source_runtime_version",
    "contract_version",
)
P3_SOURCE_EVENT_NULLABLE_FIELDS = frozenset(
    {
        "start_ms",
        "end_ms",
        "actor_track_id",
        "secondary_track_id",
        "actor_game_player_id",
        "secondary_game_player_id",
        "team_id",
        "location_court_m",
        "source_runtime_version",
    }
)


class P3ContractValidationError(ValueError):
    """Raised when data would cross the P2 -> P3 boundary ambiguously or lossy."""


class DerivationState(str, Enum):
    """Lifecycle state for reprocessable P3 derived data."""

    CURRENT = "current"
    STALE = "stale"
    SUPERSEDED = "superseded"
    FAILED = "failed"


def _required_text(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise P3ContractValidationError(f"{field_name} must be a non-empty string")
    return text


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _bounded_probability(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise P3ContractValidationError(f"{field_name} must be a number in [0, 1]")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise P3ContractValidationError(
            f"{field_name} must be a number in [0, 1]"
        ) from exc
    if not isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise P3ContractValidationError(f"{field_name} must be a finite number in [0, 1]")
    return parsed


def _integer(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise P3ContractValidationError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise P3ContractValidationError(f"{field_name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise P3ContractValidationError(f"{field_name} must be an integer")
    if minimum is not None and parsed < minimum:
        raise P3ContractValidationError(f"{field_name} must be >= {minimum}")
    return parsed


def _optional_integer(
    value: Any,
    field_name: str,
    *,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name, minimum=minimum)


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise P3ContractValidationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise P3ContractValidationError(f"{path} keys must be strings")
            _validate_json_value(child, f"{path}.{key}")
        return
    raise P3ContractValidationError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise P3ContractValidationError(f"{field_name} must be a JSON object")
    _validate_json_value(value, field_name)
    return dict(value)


def require_supported_p3_source_contract(version: Any) -> str:
    normalized = _required_text(version, "contract_version")
    if normalized != P3_SOURCE_EVENT_CONTRACT_VERSION:
        raise P3ContractValidationError(
            "unsupported P3 source-event contract version "
            f"{normalized!r}; expected {P3_SOURCE_EVENT_CONTRACT_VERSION!r}"
        )
    return normalized


def p3_source_event_contract_manifest() -> dict[str, Any]:
    """Return the machine-readable frozen v1 boundary contract."""

    return {
        "contract_version": P3_SOURCE_EVENT_CONTRACT_VERSION,
        "fields": list(P3_SOURCE_EVENT_FIELDS),
        "nullable_fields": sorted(P3_SOURCE_EVENT_NULLABLE_FIELDS),
        "supported_event_types": sorted(P3_SUPPORTED_SOURCE_EVENT_TYPES),
        "supported_decisions": sorted(P3_SUPPORTED_SOURCE_DECISIONS),
        "canonical_time_unit": "source_video_milliseconds",
        "unknown_representation": "null",
        "confidence_semantics": "source confidence and uncertainty remain separate",
    }


@dataclass(slots=True)
class P3SourceEvent:
    """Frozen v1, loss-minimizing boundary between P2 events and P3.

    The object validates rather than repairs source data. Unknown evidence is represented
    explicitly with ``None``; invalid values are rejected instead of being clamped or
    converted into plausible-looking basketball truth.
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
        self.source_event_id = _required_text(self.source_event_id, "source_event_id")
        self.game_id = _required_text(self.game_id, "game_id")
        self.analysis_run_id = _required_text(self.analysis_run_id, "analysis_run_id")
        self.event_type = _required_text(self.event_type, "event_type")
        if self.event_type not in P3_SUPPORTED_SOURCE_EVENT_TYPES:
            raise P3ContractValidationError(
                f"unsupported P3 source event_type {self.event_type!r}"
            )
        self.decision = _required_text(self.decision, "decision")
        if self.decision not in P3_SUPPORTED_SOURCE_DECISIONS:
            raise P3ContractValidationError(
                f"unsupported P3 source decision {self.decision!r}"
            )

        self.confidence = _bounded_probability(self.confidence, "confidence")
        self.uncertainty = _bounded_probability(self.uncertainty, "uncertainty")
        self.frame_start = _integer(self.frame_start, "frame_start", minimum=0)
        self.frame_end = _integer(self.frame_end, "frame_end", minimum=0)
        if self.frame_end < self.frame_start:
            raise P3ContractValidationError("frame_end must be >= frame_start")

        self.start_ms = _optional_integer(self.start_ms, "start_ms", minimum=0)
        self.end_ms = _optional_integer(self.end_ms, "end_ms", minimum=0)
        if self.start_ms is not None and self.end_ms is not None and self.end_ms < self.start_ms:
            raise P3ContractValidationError("end_ms must be >= start_ms")

        self.actor_track_id = _optional_integer(
            self.actor_track_id, "actor_track_id", minimum=0
        )
        self.secondary_track_id = _optional_integer(
            self.secondary_track_id, "secondary_track_id", minimum=0
        )
        self.actor_game_player_id = _optional_text(
            self.actor_game_player_id, "actor_game_player_id"
        )
        self.secondary_game_player_id = _optional_text(
            self.secondary_game_player_id, "secondary_game_player_id"
        )
        self.team_id = _optional_text(self.team_id, "team_id")

        if self.location_court_m is not None:
            if not isinstance(self.location_court_m, (list, tuple)) or len(self.location_court_m) != 2:
                raise P3ContractValidationError(
                    "location_court_m must be null or exactly [x, y]"
                )
            x = float(self.location_court_m[0])
            y = float(self.location_court_m[1])
            if not isfinite(x) or not isfinite(y):
                raise P3ContractValidationError(
                    "location_court_m coordinates must be finite"
                )
            self.location_court_m = (x, y)

        if not isinstance(self.evidence, (list, tuple)):
            raise P3ContractValidationError("evidence must be an array of non-empty strings")
        normalized_evidence: list[str] = []
        for index, item in enumerate(self.evidence):
            text = _required_text(item, f"evidence[{index}]")
            normalized_evidence.append(text)
        self.evidence = tuple(normalized_evidence)

        self.attributes = _json_object(self.attributes, "attributes")
        self.provenance = _json_object(self.provenance, "provenance")
        self.source_contract_version = _required_text(
            self.source_contract_version, "source_contract_version"
        )
        self.source_runtime_version = _optional_text(
            self.source_runtime_version, "source_runtime_version"
        )
        self.contract_version = require_supported_p3_source_contract(
            self.contract_version
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "P3SourceEvent":
        """Parse an exact v1 normalized payload without silently filling fields."""

        if not isinstance(payload, dict):
            raise P3ContractValidationError("P3 source-event payload must be an object")
        missing = [field for field in P3_SOURCE_EVENT_FIELDS if field not in payload]
        if missing:
            raise P3ContractValidationError(
                "P3 source-event payload is missing fields: " + ", ".join(missing)
            )
        extras = sorted(set(payload) - set(P3_SOURCE_EVENT_FIELDS))
        if extras:
            raise P3ContractValidationError(
                "P3 source-event v1 payload contains unknown fields: "
                + ", ".join(extras)
            )
        return cls(**{field: payload[field] for field in P3_SOURCE_EVENT_FIELDS})

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


def validate_source_event_batch(
    events: Iterable[P3SourceEvent],
    *,
    game_id: str | None = None,
    analysis_run_id: str | None = None,
) -> tuple[P3SourceEvent, ...]:
    """Validate cross-event invariants for one P3 materialization input batch."""

    rows = tuple(events)
    if not rows:
        return rows

    expected_game = _required_text(game_id, "game_id") if game_id is not None else rows[0].game_id
    expected_run = (
        _required_text(analysis_run_id, "analysis_run_id")
        if analysis_run_id is not None
        else rows[0].analysis_run_id
    )

    event_ids: set[str] = set()
    source_contract_versions: set[str] = set()
    runtime_versions: set[str] = set()
    for event in rows:
        require_supported_p3_source_contract(event.contract_version)
        if event.game_id != expected_game:
            raise P3ContractValidationError(
                "P3 source-event batch cannot mix game_id values"
            )
        if event.analysis_run_id != expected_run:
            raise P3ContractValidationError(
                "P3 source-event batch cannot mix analysis_run_id values"
            )
        if event.source_event_id in event_ids:
            raise P3ContractValidationError(
                f"duplicate source_event_id in P3 batch: {event.source_event_id!r}"
            )
        event_ids.add(event.source_event_id)
        source_contract_versions.add(event.source_contract_version)
        if event.source_runtime_version is not None:
            runtime_versions.add(event.source_runtime_version)

    if len(source_contract_versions) > 1:
        raise P3ContractValidationError(
            "P3 source-event batch cannot mix P2 source contract versions"
        )
    if len(runtime_versions) > 1:
        raise P3ContractValidationError(
            "P3 source-event batch cannot mix known P2 runtime versions"
        )
    return rows


@dataclass(slots=True)
class P3DerivationSpec:
    """Version/config identity for a reprocessable P3 derived dataset."""

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
        self.game_id = _required_text(self.game_id, "game_id")
        self.analysis_run_id = _required_text(self.analysis_run_id, "analysis_run_id")
        self.derivation_key = _required_text(self.derivation_key, "derivation_key")
        self.derivation_version = _required_text(
            self.derivation_version, "derivation_version"
        )
        self.configuration_version = _required_text(
            self.configuration_version, "configuration_version"
        )
        self.source_contract_version = _required_text(
            self.source_contract_version, "source_contract_version"
        )
        self.contract_version = _required_text(self.contract_version, "contract_version")
        if self.contract_version != P3_DERIVATION_CONTRACT_VERSION:
            raise P3ContractValidationError(
                "unsupported P3 derivation contract version "
                f"{self.contract_version!r}; expected {P3_DERIVATION_CONTRACT_VERSION!r}"
            )
        if not isinstance(self.source_versions, dict):
            raise P3ContractValidationError("source_versions must be an object")
        self.source_versions = {
            _required_text(key, "source_versions key"): _required_text(
                value, f"source_versions[{key!r}]"
            )
            for key, value in self.source_versions.items()
        }
        self.configuration = _json_object(self.configuration, "configuration")

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
            allow_nan=False,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {**self.fingerprint_payload(), "fingerprint": self.fingerprint()}


def needs_reprocessing(previous_fingerprint: str | None, spec: P3DerivationSpec) -> bool:
    """Return True when a missing/different derivation fingerprint must be rebuilt."""

    return not previous_fingerprint or str(previous_fingerprint) != spec.fingerprint()
