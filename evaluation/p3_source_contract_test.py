from __future__ import annotations

from core.basketball_event import BasketballEvent, BasketballEventType, EventDecision
from core.p3_contract import P3DerivationSpec, needs_reprocessing
from services.p3.p2_adapter import adapt_basketball_event, adapt_basketball_events


def _event() -> BasketballEvent:
    return BasketballEvent(
        event_id="turnover-candidate:100:104",
        event_type=BasketballEventType.TURNOVER_CANDIDATE,
        frame_start=100,
        frame_end=104,
        timestamp_start_sec=3.333,
        timestamp_end_sec=3.466,
        confidence=0.73,
        uncertainty=0.27,
        decision=EventDecision.CANDIDATE,
        actor_track_id=12,
        secondary_track_id=21,
        team_id="team-a",
        location_court_m=(11.2, 4.8),
        evidence=("possession_change", "team_transition"),
        provenance={"version": "possession_tracker_v1", "frame_id": 104},
        attributes={"from_team_id": "team-a", "to_team_id": "team-b"},
    )


def test_adapter_preserves_p2_uncertainty_and_source_evidence() -> None:
    source = _event()
    adapted = adapt_basketball_event(
        source,
        game_id="game-1",
        analysis_run_id="run-1",
        p2_runtime_version="p2-engine-v1",
    )

    assert adapted.source_event_id == source.event_id
    assert adapted.event_type == "turnover_candidate"
    assert adapted.decision == "candidate"
    assert adapted.confidence == 0.73
    assert adapted.uncertainty == 0.27
    assert adapted.start_ms == 3333
    assert adapted.end_ms == 3466
    assert adapted.actor_track_id == 12
    assert adapted.actor_game_player_id is None
    assert adapted.evidence == source.evidence
    assert adapted.provenance == source.provenance
    assert adapted.source_runtime_version == "p2-engine-v1"


def test_adapter_adds_game_player_identity_only_when_resolver_knows_it() -> None:
    mapping = {12: "game_player_03"}
    adapted = adapt_basketball_event(
        _event(),
        game_id="game-1",
        analysis_run_id="run-1",
        resolve_game_player=mapping.get,
    )

    assert adapted.actor_game_player_id == "game_player_03"
    assert adapted.secondary_game_player_id is None
    assert adapted.actor_track_id == 12
    assert adapted.secondary_track_id == 21


def test_batch_adapter_preserves_source_order() -> None:
    first = _event()
    second = BasketballEvent(
        event_id="shot-attempt:120:125",
        event_type=BasketballEventType.SHOT_ATTEMPT,
        frame_start=120,
        frame_end=125,
        timestamp_start_sec=4.0,
        timestamp_end_sec=4.166,
        confidence=0.61,
        uncertainty=0.39,
        decision=EventDecision.CANDIDATE,
    )
    adapted = adapt_basketball_events(
        [first, second],
        game_id="game-1",
        analysis_run_id="run-1",
    )
    assert [event.source_event_id for event in adapted] == [first.event_id, second.event_id]


def test_derivation_fingerprint_is_deterministic_and_configuration_sensitive() -> None:
    spec_a = P3DerivationSpec(
        game_id="game-1",
        analysis_run_id="run-1",
        derivation_key="review_items",
        derivation_version="review-items-v1",
        configuration_version="config-v1",
        source_versions={"p2": "p2-v1", "p1": "p1-v1"},
        configuration={"min_confidence": 0.50, "pre_roll_ms": 6000},
    )
    spec_b = P3DerivationSpec(
        game_id="game-1",
        analysis_run_id="run-1",
        derivation_key="review_items",
        derivation_version="review-items-v1",
        configuration_version="config-v1",
        source_versions={"p1": "p1-v1", "p2": "p2-v1"},
        configuration={"pre_roll_ms": 6000, "min_confidence": 0.50},
    )
    changed = P3DerivationSpec(
        game_id="game-1",
        analysis_run_id="run-1",
        derivation_key="review_items",
        derivation_version="review-items-v1",
        configuration_version="config-v2",
        source_versions={"p1": "p1-v1", "p2": "p2-v2"},
        configuration={"pre_roll_ms": 6000, "min_confidence": 0.65},
    )

    assert spec_a.fingerprint() == spec_b.fingerprint()
    assert spec_a.fingerprint() != changed.fingerprint()
    assert needs_reprocessing(spec_a.fingerprint(), spec_b) is False
    assert needs_reprocessing(spec_a.fingerprint(), changed) is True
    assert needs_reprocessing(None, spec_a) is True


def main() -> None:
    test_adapter_preserves_p2_uncertainty_and_source_evidence()
    test_adapter_adds_game_player_identity_only_when_resolver_knows_it()
    test_batch_adapter_preserves_source_order()
    test_derivation_fingerprint_is_deterministic_and_configuration_sensitive()
    print("p3_source_contract_test: PASS")


if __name__ == "__main__":
    main()
