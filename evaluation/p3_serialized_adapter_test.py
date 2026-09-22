from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from services.p3.materializer import load_p2_event_rows
from services.p3.p2_adapter import adapt_p2_event_row


def test_serialized_p2_event_preserves_source_semantics() -> None:
    event = adapt_p2_event_row(
        {
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
            "actor_game_player_id": "game_player_03",
            "secondary_game_player_id": "game_player_08",
            "location_court_m": [8.5, 6.0],
            "evidence": ["release", "receiver"],
            "provenance": {"version": "possession_tracker_v1"},
            "attributes": {"flight_supported": True},
        },
        game_id="game-1",
        analysis_run_id="run-1",
        p2_runtime_version="p2_engine_v1",
    )

    assert event.source_event_id == "pass-1"
    assert event.event_type == "pass_candidate"
    assert event.decision == "candidate"
    assert event.start_ms == 3250
    assert event.end_ms == 3520
    assert event.actor_track_id == 4
    assert event.secondary_track_id == 9
    assert event.actor_game_player_id == "game_player_03"
    assert event.secondary_game_player_id == "game_player_08"
    assert event.team_id == "home"
    assert event.location_court_m == (8.5, 6.0)
    assert event.source_runtime_version == "p2_engine_v1"


def test_jsonl_loader_rejects_non_object_rows() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        path.write_text(
            json.dumps({"event_id": "one"}) + "\n" + json.dumps(["bad"]) + "\n",
            encoding="utf-8",
        )
        try:
            load_p2_event_rows(path)
        except ValueError as exc:
            assert "line 2" in str(exc)
            assert "must contain an object" in str(exc)
        else:
            raise AssertionError("non-object JSONL rows must be rejected")


def test_serialized_adapter_rejects_missing_identity_fields() -> None:
    try:
        adapt_p2_event_row(
            {
                "event_type": "shot_attempt",
                "decision": "candidate",
            },
            game_id="game-1",
            analysis_run_id="run-1",
        )
    except ValueError as exc:
        assert "missing event_id" in str(exc)
    else:
        raise AssertionError("event_id is required for traceability")