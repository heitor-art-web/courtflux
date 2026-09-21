from core.basketball_event import BasketballEvent, BasketballEventType, EventDecision
from services.p2.engine_v1 import P2EngineV1


def _event(
    event_type: BasketballEventType,
    *,
    actor: int,
    secondary: int,
    team: str | None,
    attributes: dict | None = None,
) -> BasketballEvent:
    return BasketballEvent(
        event_id=f"test:{event_type.value}:{actor}:{secondary}",
        event_type=event_type,
        frame_start=100,
        frame_end=103,
        timestamp_start_sec=1.0,
        timestamp_end_sec=1.1,
        confidence=0.9,
        uncertainty=0.1,
        decision=EventDecision.CANDIDATE,
        actor_track_id=actor,
        secondary_track_id=secondary,
        team_id=team,
        attributes=dict(attributes or {}),
    )


def test_turnover_reclassifies_when_both_fresh_endpoints_agree() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=101,
        team_hints={11: ("team_a", 0.94), 12: ("team_a", 0.92)},
    )
    event = _event(
        BasketballEventType.TURNOVER_CANDIDATE,
        actor=11,
        secondary=12,
        team="team_b",
    )

    result = engine._normalize_event_team_publication(event, frame_id=103)

    assert result.event_type is BasketballEventType.PASS_CANDIDATE
    assert result.team_id == "team_a"
    assert result.attributes["event_type_before_publication"] == "turnover_candidate"
    assert result.attributes["team_attribution_publication_reason"] == (
        "fresh_same_team_endpoints_reclassified_turnover_to_pass"
    )


def test_turnover_does_not_reclassify_when_fresh_endpoints_disagree() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=101,
        team_hints={11: ("team_a", 0.94), 12: ("team_b", 0.92)},
    )
    event = _event(
        BasketballEventType.TURNOVER_CANDIDATE,
        actor=11,
        secondary=12,
        team="team_a",
    )

    result = engine._normalize_event_team_publication(event, frame_id=103)

    assert result.event_type is BasketballEventType.TURNOVER_CANDIDATE
    assert result.team_id == "team_a"


def test_confirmed_same_team_switch_with_release_interval_emits_pass() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=101,
        team_hints={21: ("team_a", 0.95), 22: ("team_a", 0.93)},
    )
    change = _event(
        BasketballEventType.POSSESSION_CHANGE,
        actor=22,
        secondary=21,
        team="team_a",
        attributes={
            "flight_supported": False,
            "flight_frames": 4,
            "flight_direct_observations": 1,
            "flight_predicted_observations": 1,
            "flight_displacement_player_heights": 0.2,
        },
    )

    result = engine._normalize_and_recover_transition_events([change], frame_id=103)

    assert len(result) == 2
    recovered = result[1]
    assert recovered.event_type is BasketballEventType.PASS_CANDIDATE
    assert recovered.actor_track_id == 21
    assert recovered.secondary_track_id == 22
    assert recovered.team_id == "team_a"
    assert recovered.attributes["team_attribution_publication_reason"] == (
        "confirmed_same_team_switch_release_interval"
    )


def test_confirmed_switch_without_fresh_endpoint_agreement_does_not_emit_pass() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=101,
        team_hints={21: ("team_a", 0.95), 22: ("team_b", 0.93)},
    )
    change = _event(
        BasketballEventType.POSSESSION_CHANGE,
        actor=22,
        secondary=21,
        team="team_b",
        attributes={
            "flight_supported": False,
            "flight_frames": 4,
            "flight_direct_observations": 1,
            "flight_predicted_observations": 1,
        },
    )

    result = engine._normalize_and_recover_transition_events([change], frame_id=103)

    assert len(result) == 1
    assert result[0].event_type is BasketballEventType.POSSESSION_CHANGE


def main() -> None:
    test_turnover_reclassifies_when_both_fresh_endpoints_agree()
    test_turnover_does_not_reclassify_when_fresh_endpoints_disagree()
    test_confirmed_same_team_switch_with_release_interval_emits_pass()
    test_confirmed_switch_without_fresh_endpoint_agreement_does_not_emit_pass()
    print("p2_transition_team_resolution_test: PASS")


if __name__ == "__main__":
    main()
