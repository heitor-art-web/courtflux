from __future__ import annotations

from collections.abc import Callable, Iterable
from math import isfinite
from typing import Any

from core.basketball_event import BasketballEvent
from core.p3_contract import (
    P3ContractValidationError,
    P3SourceEvent,
    validate_source_event_batch,
)


P2_BASKETBALL_EVENT_CONTRACT_VERSION = "core.basketball_event.v1"
P2_CANONICAL_EVENT_ROW_FIELDS = (
    "event_id",
    "event_type",
    "frame_start",
    "frame_end",
    "timestamp_start_sec",
    "timestamp_end_sec",
    "confidence",
    "uncertainty",
    "decision",
    "actor_track_id",
    "secondary_track_id",
    "team_id",
    "location_court_m",
    "evidence",
    "provenance",
    "attributes",
)

GamePlayerResolver = Callable[[int], str | None]


def _seconds_to_ms(value: float | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise P3ContractValidationError("source timestamp must be a finite number or null")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise P3ContractValidationError(
            "source timestamp must be a finite number or null"
        ) from exc
    if not isfinite(seconds) or seconds < 0:
        raise P3ContractValidationError(
            "source timestamp must be a non-negative finite number or null"
        )
    return int(round(seconds * 1000.0))


def _resolve_game_player(
    track_id: int | None,
    resolver: GamePlayerResolver | None,
) -> str | None:
    if track_id is None or resolver is None:
        return None
    resolved = resolver(int(track_id))
    return str(resolved) if resolved is not None else None


def _require_canonical_row_fields(row: dict[str, Any]) -> None:
    if "event_id" not in row or not str(row.get("event_id") or "").strip():
        raise P3ContractValidationError("serialized P2 event is missing event_id")
    missing = [field for field in P2_CANONICAL_EVENT_ROW_FIELDS if field not in row]
    if missing:
        event_id = str(row["event_id"])
        raise P3ContractValidationError(
            f"serialized P2 event {event_id!r} is missing canonical fields: "
            + ", ".join(missing)
        )


def adapt_basketball_event(
    event: BasketballEvent,
    *,
    game_id: str,
    analysis_run_id: str,
    resolve_game_player: GamePlayerResolver | None = None,
    p2_contract_version: str = P2_BASKETBALL_EVENT_CONTRACT_VERSION,
    p2_runtime_version: str | None = None,
) -> P3SourceEvent:
    """Normalize one canonical P2 event into the frozen P3 source contract.

    This adapter is intentionally loss-minimizing. It does not apply P3 thresholds,
    promote candidates, alter confidence, infer missing players, or inspect P2 engine
    internals. Invalid source data is rejected instead of repaired.
    """

    return P3SourceEvent(
        source_event_id=event.event_id,
        game_id=game_id,
        analysis_run_id=analysis_run_id,
        event_type=event.event_type.value,
        decision=event.decision.value,
        confidence=event.confidence,
        uncertainty=event.uncertainty,
        frame_start=event.frame_start,
        frame_end=event.frame_end,
        start_ms=_seconds_to_ms(event.timestamp_start_sec),
        end_ms=_seconds_to_ms(event.timestamp_end_sec),
        actor_track_id=event.actor_track_id,
        secondary_track_id=event.secondary_track_id,
        actor_game_player_id=_resolve_game_player(event.actor_track_id, resolve_game_player),
        secondary_game_player_id=_resolve_game_player(
            event.secondary_track_id, resolve_game_player
        ),
        team_id=event.team_id,
        location_court_m=event.location_court_m,
        evidence=event.evidence,
        attributes=event.attributes,
        provenance=event.provenance,
        source_contract_version=p2_contract_version,
        source_runtime_version=p2_runtime_version,
    )


def adapt_p2_event_row(
    row: dict[str, Any],
    *,
    game_id: str,
    analysis_run_id: str,
    resolve_game_player: GamePlayerResolver | None = None,
    p2_contract_version: str = P2_BASKETBALL_EVENT_CONTRACT_VERSION,
    p2_runtime_version: str | None = None,
) -> P3SourceEvent:
    """Adapt one serialized canonical P2 event without reconstructing P2 internals."""

    if not isinstance(row, dict):
        raise P3ContractValidationError("serialized P2 event must be an object")
    _require_canonical_row_fields(row)

    event_id = str(row["event_id"] or "").strip()
    event_type = str(row["event_type"] or "").strip()
    decision = str(row["decision"] or "").strip()
    if not event_type:
        raise P3ContractValidationError(
            f"serialized P2 event {event_id!r} is missing event_type"
        )
    if decision not in {"candidate", "confirmed"}:
        raise P3ContractValidationError(
            f"serialized P2 event {event_id!r} has unsupported decision {decision!r}"
        )

    actor_track_id = row["actor_track_id"]
    secondary_track_id = row["secondary_track_id"]
    actor_track_id = None if actor_track_id is None else int(actor_track_id)
    secondary_track_id = (
        None if secondary_track_id is None else int(secondary_track_id)
    )

    location = row["location_court_m"]
    location_court_m: tuple[float, float] | None = None
    if location is not None:
        if not isinstance(location, (list, tuple)) or len(location) != 2:
            raise P3ContractValidationError(
                f"serialized P2 event {event_id!r} has invalid location_court_m"
            )
        location_court_m = (float(location[0]), float(location[1]))

    evidence = row["evidence"]
    if not isinstance(evidence, (list, tuple)):
        raise P3ContractValidationError(
            f"serialized P2 event {event_id!r} evidence must be an array"
        )
    attributes = row["attributes"]
    provenance = row["provenance"]
    if not isinstance(attributes, dict):
        raise P3ContractValidationError(
            f"serialized P2 event {event_id!r} attributes must be an object"
        )
    if not isinstance(provenance, dict):
        raise P3ContractValidationError(
            f"serialized P2 event {event_id!r} provenance must be an object"
        )

    team_id = row["team_id"]
    if team_id in {None, "", "unknown", "UNKNOWN"}:
        team_id = None

    return P3SourceEvent(
        source_event_id=event_id,
        game_id=str(game_id),
        analysis_run_id=str(analysis_run_id),
        event_type=event_type,
        decision=decision,
        confidence=row["confidence"],
        uncertainty=row["uncertainty"],
        frame_start=row["frame_start"],
        frame_end=row["frame_end"],
        start_ms=_seconds_to_ms(row["timestamp_start_sec"]),
        end_ms=_seconds_to_ms(row["timestamp_end_sec"]),
        actor_track_id=actor_track_id,
        secondary_track_id=secondary_track_id,
        actor_game_player_id=(
            str(row["actor_game_player_id"])
            if row.get("actor_game_player_id") is not None
            else _resolve_game_player(actor_track_id, resolve_game_player)
        ),
        secondary_game_player_id=(
            str(row["secondary_game_player_id"])
            if row.get("secondary_game_player_id") is not None
            else _resolve_game_player(secondary_track_id, resolve_game_player)
        ),
        team_id=None if team_id is None else str(team_id),
        location_court_m=location_court_m,
        evidence=tuple(evidence),
        attributes=dict(attributes),
        provenance=dict(provenance),
        source_contract_version=str(p2_contract_version),
        source_runtime_version=(
            str(p2_runtime_version) if p2_runtime_version is not None else None
        ),
    )


def adapt_basketball_events(
    events: Iterable[BasketballEvent],
    *,
    game_id: str,
    analysis_run_id: str,
    resolve_game_player: GamePlayerResolver | None = None,
    p2_contract_version: str = P2_BASKETBALL_EVENT_CONTRACT_VERSION,
    p2_runtime_version: str | None = None,
) -> list[P3SourceEvent]:
    """Adapt events in source order so P3 never creates an alternate event timeline."""

    adapted = [
        adapt_basketball_event(
            event,
            game_id=game_id,
            analysis_run_id=analysis_run_id,
            resolve_game_player=resolve_game_player,
            p2_contract_version=p2_contract_version,
            p2_runtime_version=p2_runtime_version,
        )
        for event in events
    ]
    validate_source_event_batch(
        adapted,
        game_id=game_id,
        analysis_run_id=analysis_run_id,
    )
    return adapted


def adapt_p2_event_rows(
    rows: Iterable[dict[str, Any]],
    *,
    game_id: str,
    analysis_run_id: str,
    resolve_game_player: GamePlayerResolver | None = None,
    p2_contract_version: str = P2_BASKETBALL_EVENT_CONTRACT_VERSION,
    p2_runtime_version: str | None = None,
) -> list[P3SourceEvent]:
    """Adapt serialized P2 rows in source order with the same frozen contract."""

    adapted = [
        adapt_p2_event_row(
            row,
            game_id=game_id,
            analysis_run_id=analysis_run_id,
            resolve_game_player=resolve_game_player,
            p2_contract_version=p2_contract_version,
            p2_runtime_version=p2_runtime_version,
        )
        for row in rows
    ]
    validate_source_event_batch(
        adapted,
        game_id=game_id,
        analysis_run_id=analysis_run_id,
    )
    return adapted
