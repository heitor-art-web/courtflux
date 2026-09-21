from core.basketball_event import BasketballEvent, BasketballEventType
from services.p2.engine_v1 import P2EngineV1


def _event(
    event_type: BasketballEventType,
    *,
    team_id: str | None,
    actor_track_id: int = 11,
    secondary_track_id: int | None = 12,
) -> BasketballEvent:
    return BasketballEvent(
        event_id=f"test:{event_type.value}",
        event_type=event_type,
        frame_start=100,
        frame_end=103,
        timestamp_start_sec=1.0,
        timestamp_end_sec=1.1,
        confidence=0.9,
        uncertainty=0.1,
        actor_track_id=actor_track_id,
        secondary_track_id=secondary_track_id,
        team_id=team_id,
    )


def test_pass_memory_without_recent_fresh_endpoint_support_abstains() -> None:
    engine = P2EngineV1()
    event = _event(BasketballEventType.PASS_CANDIDATE, team_id="team_b")

    result = engine._normalize_event_team_publication(event, frame_id=103)

    assert result.team_id is None
    assert result.attributes["internal_team_id_before_publication"] == "team_b"
    assert result.attributes["team_attribution_publication_reason"] == (
        "no_recent_fresh_endpoint_team_support"
    )


def test_pass_recent_fresh_receiver_support_keeps_team() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=100,
        team_hints={12: ("team_b", 0.94)},
    )
    event = _event(BasketballEventType.PASS_CANDIDATE, team_id="team_b")

    result = engine._normalize_event_team_publication(event, frame_id=103)

    assert result.team_id == "team_b"
    assert result.attributes["team_attribution_publication_reason"] == "fresh_endpoint_support"
    assert result.attributes["team_attribution_fresh_hint_tracks"] == [12]


def test_pass_conflicting_fresh_endpoint_support_abstains() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=101,
        team_hints={11: ("team_a", 0.95), 12: ("team_b", 0.93)},
    )
    event = _event(BasketballEventType.PASS_CANDIDATE, team_id="team_b")

    result = engine._normalize_event_team_publication(event, frame_id=103)

    assert result.team_id is None
    assert result.attributes["team_attribution_publication_reason"] == (
        "fresh_endpoint_team_conflict"
    )


def test_stale_fresh_hint_does_not_publish_pass_team() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=90,
        team_hints={12: ("team_b", 0.94)},
    )
    event = _event(BasketballEventType.PASS_CANDIDATE, team_id="team_b")

    result = engine._normalize_event_team_publication(event, frame_id=100)

    assert result.team_id is None


def test_turnover_requires_fresh_actor_support_not_receiver_support() -> None:
    engine = P2EngineV1()
    engine._observe_event_team_hints(
        frame_id=100,
        team_hints={12: ("team_b", 0.95)},
    )
    event = _event(BasketballEventType.TURNOVER_CANDIDATE, team_id="team_a")

    result = engine._normalize_event_team_publication(event, frame_id=102)

    assert result.team_id is None


def test_shot_team_uses_public_supported_owner_not_internal_stale_memory() -> None:
    engine = P2EngineV1()
    tracker = engine.possession_tracker
    tracker.current_track_id = 40
    tracker.current_team_id = "team_a"
    tracker.current_confidence = 0.8
    tracker._lost_frames = 1
    tracker._raw_best_track_id = 41
    tracker._raw_best_score = 0.7

    track_id, team_id, source = engine._pre_release_public_attribution(frame_id=100)

    assert track_id == 40
    assert team_id is None
    assert source == "pre_release_possession_public_abstained"
    assert tracker.current_team_id == "team_a"


def test_shot_team_keeps_public_supported_owner_team() -> None:
    engine = P2EngineV1()
    tracker = engine.possession_tracker
    tracker.current_track_id = 40
    tracker.current_team_id = "team_b"
    tracker.current_confidence = 0.8
    tracker._lost_frames = 0
    tracker._candidate_track_id = None
    tracker._candidate_frames = 0

    track_id, team_id, source = engine._pre_release_public_attribution(frame_id=100)

    assert track_id == 40
    assert team_id == "team_b"
    assert source == "pre_release_possession_public_team"


def main() -> None:
    test_pass_memory_without_recent_fresh_endpoint_support_abstains()
    test_pass_recent_fresh_receiver_support_keeps_team()
    test_pass_conflicting_fresh_endpoint_support_abstains()
    test_stale_fresh_hint_does_not_publish_pass_team()
    test_turnover_requires_fresh_actor_support_not_receiver_support()
    test_shot_team_uses_public_supported_owner_not_internal_stale_memory()
    test_shot_team_keeps_public_supported_owner_team()
    print("p2_event_team_publication_test: PASS")


if __name__ == "__main__":
    main()
