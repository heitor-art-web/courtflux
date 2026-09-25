from __future__ import annotations

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import (
    BasketballEvent,
    BasketballEventType,
    EventDecision,
    ShotOutcome,
)
from core.court_state import CourtState
from services.p2.engine_v6_batch import P2EngineV6Batch
from services.p2.shot_event_tracker_v1_3 import ShotEventTrackerV13
from services.vision.court_state_gameplay_v3 import CourtStateEstimatorV3
from services.vision.gate_d_limited_scope_v6 import select_limited_bootstrap_candidates


def _shot_event(**attributes) -> BasketballEvent:
    return BasketballEvent(
        event_id="shot-test",
        event_type=BasketballEventType.SHOT_ATTEMPT,
        frame_start=10,
        frame_end=20,
        timestamp_start_sec=1.0,
        timestamp_end_sec=1.2,
        confidence=0.45,
        uncertainty=0.55,
        decision=EventDecision.CANDIDATE,
        evidence=(),
        provenance={},
        attributes={"outcome": "unknown", **attributes},
    )


def test_v6_a_hysteresis_contract() -> None:
    estimator = CourtStateEstimatorV3()
    assert estimator.non_gameplay_enter_sec == 3.0
    assert estimator.weak_live_hold_sec == 1.0
    assert estimator.gameplay_resume_sec == 0.20


def test_v6_b_ft_single_mouth_plus_descent_confirms_made() -> None:
    reason = ShotEventTrackerV13._deferred_confirmation_reason(
        {
            "rim_observations": 6,
            "near_x_observations": 1,
            "inside_mouth_observations": 1,
            "below_rim_observations": 6,
            "descending_steps": 4,
            "ascending_steps": 2,
            "incoherent_rim_steps": 0,
            "min_abs_x_offset_rim_widths": 0.0103,
        }
    )
    assert reason == "deferred_rim_exit_mouth_below_descent_confirmed"


def test_v6_b_incoherent_deferred_geometry_still_abstains() -> None:
    reason = ShotEventTrackerV13._deferred_confirmation_reason(
        {
            "rim_observations": 6,
            "near_x_observations": 1,
            "inside_mouth_observations": 1,
            "below_rim_observations": 6,
            "descending_steps": 4,
            "incoherent_rim_steps": 1,
            "min_abs_x_offset_rim_widths": 0.01,
        }
    )
    assert reason is None


def test_v6_b_unconfirmed_trajectory_start_is_not_published() -> None:
    tracker = ShotEventTrackerV13(fps=60.0)
    event = _shot_event(
        semantic_start=False,
        had_above_rim=False,
        had_rim_band=False,
        direct_ball_observations=46,
        resolution_reason="shot_window_expired_without_outcome_evidence",
        pending_bridge_used=False,
    )
    assert tracker._postprocess_public_event(event) is None
    assert tracker.diagnostics()["suppressed_public_shots"] == 1


def test_v6_b_strong_rim_expiry_becomes_miss() -> None:
    tracker = ShotEventTrackerV13(fps=60.0)
    event = _shot_event(
        semantic_start=False,
        had_above_rim=True,
        had_rim_band=True,
        direct_ball_observations=30,
        resolution_reason="shot_window_expired_without_outcome_evidence",
        pending_bridge_used=False,
    )
    result = tracker._postprocess_public_event(event)
    assert result is not None
    assert result.attributes["outcome"] == ShotOutcome.MISSED.value
    assert result.decision is EventDecision.CONFIRMED


def test_v6_b_supported_flight_can_bridge_at_rim_band() -> None:
    tracker = ShotEventTrackerV13(fps=60.0)
    tracker.set_supported_flight_hint(True)
    ball = BallState(
        frame_id=100,
        timestamp_sec=10.0,
        status=BallStateStatus.OBSERVED,
        x_px=120.0,
        y_px=115.0,
        confidence=0.9,
        uncertainty=0.1,
        source="test",
    )
    court = CourtState(
        frame_id=100,
        timestamp_sec=10.0,
        gameplay_present=True,
        gameplay_status="gameplay_analysis",
        basket_evidence=[{"kind": "rim", "bbox_px": [100.0, 100.0, 140.0, 120.0]}],
    )
    events = tracker.update(
        frame_id=100,
        timestamp_sec=10.0,
        ball_state=ball,
        court_state=court,
        semantic_evidence=[],
    )
    assert events == []
    diagnostics = tracker.diagnostics()
    assert diagnostics["active"] is True
    assert diagnostics["supported_flight_bridge_starts"] == 1


def test_v6_c_exact_actor_snapshot_can_publish_team() -> None:
    engine = P2EngineV6Batch(fps=60.0)
    engine._v6_team_evidence = {416: ("team_b", 0.97)}
    engine._v6_team_hints = {}
    team, confidence, age, source = engine._exact_actor_team_snapshot(416, frame_id=100)
    assert team == "team_b"
    assert confidence == 0.97
    assert age == 0
    assert source == "exact_actor_current_player_team"


def test_v6_d_limited_empty_scope_bootstrap_is_bounded() -> None:
    selected = select_limited_bootstrap_candidates(
        [
            {
                "track_id": 1,
                "active_on_court": None,
                "identity_eligible": True,
                "confidence": 0.93,
                "local_court_surface_ratio": 0.45,
                "court_eligibility_source": "image_space_court_surface",
                "team": "team_a",
                "team_confidence": 0.95,
            },
            {
                "track_id": 2,
                "active_on_court": None,
                "identity_eligible": True,
                "confidence": 0.90,
                "local_court_surface_ratio": 0.02,
                "court_eligibility_source": "image_space_low_surface_unknown",
                "team": "team_b",
                "team_confidence": 0.96,
            },
            {
                "track_id": 3,
                "active_on_court": False,
                "identity_eligible": True,
                "confidence": 0.99,
                "local_court_surface_ratio": 0.90,
                "court_eligibility_source": "image_space_court_surface",
                "team": "team_b",
                "team_confidence": 0.99,
            },
        ]
    )
    assert [row["track_id"] for row in selected] == [1]
    assert selected[0]["active_on_court"] is True
    assert selected[0]["gate_d_limited_bootstrap"] is True


def main() -> None:
    test_v6_a_hysteresis_contract()
    test_v6_b_ft_single_mouth_plus_descent_confirms_made()
    test_v6_b_incoherent_deferred_geometry_still_abstains()
    test_v6_b_unconfirmed_trajectory_start_is_not_published()
    test_v6_b_strong_rim_expiry_becomes_miss()
    test_v6_b_supported_flight_can_bridge_at_rim_band()
    test_v6_c_exact_actor_snapshot_can_publish_team()
    test_v6_d_limited_empty_scope_bootstrap_is_bounded()
    print("p2_v6_abcd_batch_test: PASS")


if __name__ == "__main__":
    main()
