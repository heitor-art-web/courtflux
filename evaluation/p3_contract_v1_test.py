from __future__ import annotations

from pathlib import Path

from core.basketball_event import BasketballEventType
from core.p3_contract import (
    P3ContractValidationError,
    P3SourceEvent,
    P3_SOURCE_EVENT_CONTRACT_VERSION,
    P3_SUPPORTED_SOURCE_EVENT_TYPES,
    p3_source_event_contract_manifest,
    validate_source_event_batch,
)
from services.p3.materializer import load_p2_event_rows
from services.p3.p2_adapter import adapt_p2_event_row, adapt_p2_event_rows


FIXTURE = Path(__file__).parent / "fixtures" / "p3" / "p2_events_v1.jsonl"


def _source(event_id: str = "e1", **overrides: object) -> P3SourceEvent:
    payload = {
        "source_event_id": event_id,
        "game_id": "game-1",
        "analysis_run_id": "run-1",
        "event_type": "shot_attempt",
        "decision": "candidate",
        "confidence": 0.5,
        "uncertainty": 0.5,
        "frame_start": 10,
        "frame_end": 12,
        "start_ms": 1000,
        "end_ms": 1100,
        "actor_track_id": None,
        "secondary_track_id": None,
        "actor_game_player_id": None,
        "secondary_game_player_id": None,
        "team_id": None,
        "location_court_m": None,
        "evidence": (),
        "attributes": {},
        "provenance": {},
        "source_contract_version": "core.basketball_event.v1",
        "source_runtime_version": "p2-engine-v1",
        "contract_version": P3_SOURCE_EVENT_CONTRACT_VERSION,
    }
    payload.update(overrides)
    return P3SourceEvent(**payload)


def _p2_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": "pass-1",
        "event_type": "pass_candidate",
        "frame_start": 100,
        "frame_end": 108,
        "timestamp_start_sec": 3.25,
        "timestamp_end_sec": 3.52,
        "confidence": 0.67,
        "uncertainty": 0.33,
        "decision": "candidate",
        "actor_track_id": 4,
        "secondary_track_id": 9,
        "team_id": "home",
        "location_court_m": [8.5, 6.0],
        "evidence": ["release", "receiver"],
        "provenance": {"version": "possession_tracker_v1"},
        "attributes": {"flight_supported": True},
    }
    row.update(overrides)
    return row


def test_frozen_event_type_set_matches_current_p2_enum() -> None:
    assert P3_SUPPORTED_SOURCE_EVENT_TYPES == {
        event_type.value for event_type in BasketballEventType
    }


def test_frozen_fixture_covers_every_current_p2_event_type() -> None:
    rows = load_p2_event_rows(FIXTURE)
    events = adapt_p2_event_rows(
        rows,
        game_id="fixture-game",
        analysis_run_id="fixture-run",
        p2_runtime_version="p2-engine-v1",
    )
    assert len(events) == len(BasketballEventType)
    assert {event.event_type for event in events} == P3_SUPPORTED_SOURCE_EVENT_TYPES
    assert [event.source_event_id for event in events] == [
        str(row["event_id"]) for row in rows
    ]


def test_normalized_contract_round_trips_exactly() -> None:
    event = _source(
        actor_track_id=4,
        actor_game_player_id="game_player_03",
        team_id="home",
        location_court_m=(8.2, 4.1),
        evidence=("shot_flight",),
        attributes={"outcome": "unknown"},
        provenance={"engine": "p2"},
    )
    assert P3SourceEvent.from_dict(event.as_dict()).as_dict() == event.as_dict()


def test_invalid_source_values_are_rejected_instead_of_repaired() -> None:
    for field, value in (("confidence", 1.2), ("uncertainty", -0.1)):
        try:
            _source(**{field: value})
        except P3ContractValidationError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"invalid {field} must be rejected")

    for payload, expected in (
        ({"start_ms": 1200, "end_ms": 1100}, "end_ms"),
        ({"event_type": "future_p2_event"}, "event_type"),
        ({"source_runtime_version": ""}, "source_runtime_version"),
    ):
        try:
            _source(**payload)
        except P3ContractValidationError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"invalid contract payload must reject {payload}")


def test_serialized_unknowns_must_be_explicit_not_missing() -> None:
    incomplete = _p2_row()
    del incomplete["timestamp_start_sec"]
    try:
        adapt_p2_event_row(
            incomplete,
            game_id="game-1",
            analysis_run_id="run-1",
        )
    except P3ContractValidationError as exc:
        assert "timestamp_start_sec" in str(exc)
    else:
        raise AssertionError("missing canonical fields must be rejected")

    event = adapt_p2_event_row(
        _p2_row(timestamp_start_sec=None, timestamp_end_sec=None, team_id=None),
        game_id="game-1",
        analysis_run_id="run-1",
    )
    assert event.start_ms is None
    assert event.end_ms is None
    assert event.team_id is None


def test_batch_contract_rejects_duplicate_ids_and_version_drift() -> None:
    try:
        validate_source_event_batch([_source("same"), _source("same")])
    except P3ContractValidationError as exc:
        assert "duplicate source_event_id" in str(exc)
    else:
        raise AssertionError("duplicate source-event ids must be rejected")

    try:
        validate_source_event_batch(
            [
                _source("e1", source_runtime_version="p2-a"),
                _source("e2", source_runtime_version="p2-b"),
            ]
        )
    except P3ContractValidationError as exc:
        assert "runtime versions" in str(exc)
    else:
        raise AssertionError("mixed known runtime versions must be rejected")

    try:
        validate_source_event_batch(
            [
                _source("e1", source_contract_version="core.basketball_event.v1"),
                _source("e2", source_contract_version="core.basketball_event.v2"),
            ]
        )
    except P3ContractValidationError as exc:
        assert "source contract versions" in str(exc)
    else:
        raise AssertionError("mixed P2 source contract versions must be rejected")


def test_manifest_makes_time_nullability_and_confidence_semantics_explicit() -> None:
    manifest = p3_source_event_contract_manifest()
    assert manifest["contract_version"] == P3_SOURCE_EVENT_CONTRACT_VERSION
    assert manifest["canonical_time_unit"] == "source_video_milliseconds"
    assert manifest["unknown_representation"] == "null"
    assert manifest["supported_event_types"] == sorted(P3_SUPPORTED_SOURCE_EVENT_TYPES)
    assert "source_runtime_version" in manifest["nullable_fields"]
    assert "confidence" in manifest["fields"]
    assert "uncertainty" in manifest["fields"]
