from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from core.basketball_event import BasketballEvent
from core.p3_contract import P3SourceEvent


GamePlayerResolver = Callable[[int], str | None]


def _seconds_to_ms(value: float | int | None) -> int | None:
    if value is None:
        return None
    return max(0, int(round(float(value) * 1000.0)))


def _resolve_game_player(
    track_id: int | None,
    resolver: GamePlayerResolver | None,
) -> str | None:
    if track_id is None or resolver is None:
        return None
    resolved = resolver(int(track_id))
    return str(resolved) if resolved is not None else None


def adapt_basketball_event(
    event: BasketballEvent,
    *,
    game_id: str,
    analysis_run_id: str,
    resolve_game_player: GamePlayerResolver | None = None,
    p2_contract_version: str = "core.basketball_event.v1",
    p2_runtime_version: str | None = None,
) -> P3SourceEvent:
    """Normalize one canonical P2 event into the stable P3 source contract.

    This adapter is intentionally loss-minimizing. It does not apply P3 thresholds,
    promote candidates, alter confidence, infer missing players, or inspect P2 engine
    internals. When game-level identity is unavailable the source track IDs remain the
    only identity evidence and game_player_id stays None.
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
    p2_contract_version: str = "core.basketball_event.v1",
    p2_runtime_version: str | None = None,
) -> P3SourceEvent:
    """Adapt one serialized canonical P2 event without reconstructing P2 internals."""

    event_id = str(row.get("event_id") or "").strip()
    event_type = str(row.get("event_type") or "").strip()
    decision = str(row.get("decision") or "").strip()
    if not event_id:
        raise ValueError("serialized P2 event is missing event_id")
    if not event_type:
        raise ValueError(f"serialized P2 event {event_id!r} is missing event_type")
    if decision not in {"candidate", "confirmed"}:
        raise ValueError(
            f"serialized P2 event {event_id!r} has unsupported decision {decision!r}"
        )

    actor_track_id = row.get("actor_track_id")
    secondary_track_id = row.get("secondary_track_id")
    actor_track_id = None if actor_track_id is None else int(actor_track_id)
    secondary_track_id = (
        None if secondary_track_id is None else int(secondary_track_id)
    )

    location = row.get("location_court_m")
    location_court_m: tuple[float, float] | None = None
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        location_court_m = (float(location[0]), float(location[1]))

    return P3SourceEvent(
        source_event_id=event_id,
        game_id=str(game_id),
        analysis_run_id=str(analysis_run_id),
        event_type=event_type,
        decision=decision,
        confidence=float(row.get("confidence", 0.0) or 0.0),
        uncertainty=float(row.get("uncertainty", 0.0) or 0.0),
        frame_start=int(row.get("frame_start", 0) or 0),
        frame_end=int(row.get("frame_end", row.get("frame_start", 0)) or 0),
        start_ms=_seconds_to_ms(row.get("timestamp_start_sec")),
        end_ms=_seconds_to_ms(row.get("timestamp_end_sec")),
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
        team_id=(
            None
            if row.get("team_id") in {None, "", "unknown", "UNKNOWN"}
            else str(row.get("team_id"))
        ),
        location_court_m=location_court_m,
        evidence=tuple(str(item) for item in (row.get("evidence") or []) if str(item)),
        attributes=dict(row.get("attributes") or {}),
        provenance=dict(row.get("provenance") or {}),
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
    p2_contract_version: str = "core.basketball_event.v1",
    p2_runtime_version: str | None = None,
) -> list[P3SourceEvent]:
    """Adapt events in source order so P3 never creates an alternate event timeline."""

    return [
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


def adapt_p2_event_rows(
    rows: Iterable[dict[str, Any]],
    *,
    game_id: str,
    analysis_run_id: str,
    resolve_game_player: GamePlayerResolver | None = None,
    p2_contract_version: str = "core.basketball_event.v1",
    p2_runtime_version: str | None = None,
) -> list[P3SourceEvent]:
    """Adapt serialized P2 rows in source order with the same loss-minimizing contract."""

    return [
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