from types import SimpleNamespace

from core.basketball_event import BasketballEventType
from services.p2.temporal_continuity_live_control_v1 import LiveControlP2EngineV1


def _frame(frame_id: int, timestamp: float | None = None):
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp=timestamp,
        player_states=[],
        diagnostics={},
        p2_events=[],
    )


def _p2(
    *,
    track: int | None,
    team: str | None,
    flight_active: bool,
    flight_origin: int | None = None,
    flight_start: int | None = None,
    flight_frames: int = 0,
    direct: int = 0,
    predicted: int = 0,
    events: list[dict] | None = None,
    control_state: str = "possessed",
) -> dict:
    return {
        "engine_eligible": True,
        "skipped_reason": None,
        "events": list(events or []),
        "possession": {
            "control_state": control_state,
            "current_track_id": track,
            "current_team_id": team,
            "current_confidence": 0.91,
            "team_attribution_abstained": False,
            "team_attribution_fallback_applied": False,
            "flight": {
                "active": flight_active,
                "origin_track_id": flight_origin,
                "frame_start": flight_start,
                "flight_frames": flight_frames,
                "flight_direct_observations": direct,
                "flight_predicted_observations": predicted,
                "flight_supported": direct >= 2,
            },
        },
        "shot": {"active": False},
    }


def test_recent_public_snapshot_can_attribute_ownerless_shot() -> None:
    engine = LiveControlP2EngineV1()
    engine._public_owner_snapshot = {
        "track_id": 7,
        "team_id": "team_a",
        "frame_id": 100,
        "timestamp": 3.3,
        "confidence": 0.9,
        "location_court_m": None,
        "source": "public_possession",
    }
    engine.possession_tracker.current_track_id = None

    actor, team, source = engine._pre_release_public_attribution(frame_id=105)

    assert actor == 7
    assert team is None
    assert source == "recent_public_possession_actor_only"


def test_snapshot_does_not_override_current_public_abstention() -> None:
    engine = LiveControlP2EngineV1()
    engine._public_owner_snapshot = {
        "track_id": 7,
        "team_id": "team_a",
        "frame_id": 100,
        "timestamp": 3.3,
        "confidence": 0.9,
        "location_court_m": None,
        "source": "public_possession",
    }
    engine.possession_tracker.current_track_id = 8
    engine.possession_tracker.current_team_id = None

    actor, team, source = engine._pre_release_public_attribution(frame_id=105)

    assert actor == 8
    assert team is None
    assert source == "pre_release_possession_public_abstained"


def test_release_episode_finalizes_after_same_team_receiver_lock() -> None:
    engine = LiveControlP2EngineV1()
    engine._start_or_update_release_episode(
        p2=_p2(
            track=10, team="team_a", flight_active=True,
            flight_origin=10, flight_start=99,
            flight_frames=3, direct=2,
        ),
        frame_state=_frame(100, 3.3),
    )

    terminal = {
        "event_id": "possession-change:103:10:20",
        "event_type": BasketballEventType.POSSESSION_CHANGE.value,
        "actor_track_id": 20,
        "secondary_track_id": 10,
    }
    p2 = _p2(
        track=20, team="team_a", flight_active=False,
        events=[terminal],
    )
    engine._last_frame_id = 103
    assert engine._resolve_release_episode(
        p2=p2, frame_state=_frame(103, 3.4)
    ) is None

    engine._last_frame_id = 104
    recovered = engine._resolve_release_episode(
        p2=_p2(track=20, team="team_a", flight_active=False),
        frame_state=_frame(104, 3.43),
    )

    assert recovered is not None
    assert recovered.event_type is BasketballEventType.PASS_CANDIDATE
    assert recovered.actor_track_id == 10
    assert recovered.secondary_track_id == 20
    assert recovered.team_id == "team_a"
    assert recovered.frame_start == 99


def test_conflicting_receiver_team_waits_then_resolves_same_team() -> None:
    engine = LiveControlP2EngineV1()
    engine._start_or_update_release_episode(
        p2=_p2(
            track=30, team="team_a", flight_active=True,
            flight_origin=30, flight_start=198,
            flight_frames=4, direct=1, predicted=1,
        ),
        frame_state=_frame(200, 6.6),
    )

    terminal = {
        "event_id": "possession-change:205:30:40",
        "event_type": BasketballEventType.POSSESSION_CHANGE.value,
        "actor_track_id": 40,
        "secondary_track_id": 30,
    }
    engine._last_frame_id = 205
    assert engine._resolve_release_episode(
        p2=_p2(
            track=40, team="team_b", flight_active=False,
            events=[terminal],
        ),
        frame_state=_frame(205, 6.8),
    ) is None

    engine._last_frame_id = 206
    assert engine._resolve_release_episode(
        p2=_p2(track=40, team="team_a", flight_active=False),
        frame_state=_frame(206, 6.86),
    ) is None

    engine._last_frame_id = 207
    recovered = engine._resolve_release_episode(
        p2=_p2(track=40, team="team_a", flight_active=False),
        frame_state=_frame(207, 6.9),
    )
    assert recovered is not None
    assert recovered.team_id == "team_a"


def test_explicit_pass_flight_can_resolve_unknown_origin_from_receiver() -> None:
    engine = LiveControlP2EngineV1()

    engine._start_or_update_release_episode(
        p2=_p2(
            track=None,
            team=None,
            flight_active=True,
            flight_origin=82,
            flight_start=671,
            flight_frames=28,
            direct=2,
            control_state="pass_flight",
        ),
        frame_state=_frame(698, 23.2),
    )
    assert engine._pending_release_episode is not None
    assert engine._pending_release_episode["origin_team_id"] is None

    terminal = {
        "event_id": "possession-acquired:700:114",
        "event_type": BasketballEventType.POSSESSION_ACQUIRED.value,
        "actor_track_id": 114,
        "secondary_track_id": None,
    }

    engine._last_frame_id = 700
    assert engine._resolve_release_episode(
        p2=_p2(
            track=114, team="team_a", flight_active=False,
            events=[terminal],
        ),
        frame_state=_frame(700, 23.3),
    ) is None

    # Receiver starts another flight immediately; the previous episode must survive.
    engine._start_or_update_release_episode(
        p2=_p2(
            track=114, team="team_a", flight_active=True,
            flight_origin=114, flight_start=701,
            flight_frames=1, direct=1,
        ),
        frame_state=_frame(701, 23.36),
    )
    assert engine._pending_release_episode["origin_track_id"] == 82

    engine._last_frame_id = 701
    recovered = engine._resolve_release_episode(
        p2=_p2(
            track=114, team="team_a", flight_active=True,
            flight_origin=114, flight_start=701,
            flight_frames=1, direct=1,
        ),
        frame_state=_frame(701, 23.36),
    )

    assert recovered is not None
    assert recovered.actor_track_id == 82
    assert recovered.secondary_track_id == 114
    assert recovered.team_id == "team_a"


def test_reset_clears_continuity_memory() -> None:
    engine = LiveControlP2EngineV1()
    engine._public_owner_snapshot = {"track_id": 1, "team_id": "team_a", "frame_id": 10}
    engine._pending_release_episode = {"origin_track_id": 1, "origin_team_id": "team_a"}

    engine._reset_temporal_state(reason="test", frame_id=20, timestamp=1.0)

    assert engine._public_owner_snapshot is None
    assert engine._pending_release_episode is None


def main() -> None:
    test_recent_public_snapshot_can_attribute_ownerless_shot()
    test_snapshot_does_not_override_current_public_abstention()
    test_release_episode_finalizes_after_same_team_receiver_lock()
    test_conflicting_receiver_team_waits_then_resolves_same_team()
    test_explicit_pass_flight_can_resolve_unknown_origin_from_receiver()
    test_reset_clears_continuity_memory()
    print("p2_temporal_event_continuity_test: PASS")


if __name__ == "__main__":
    main()
