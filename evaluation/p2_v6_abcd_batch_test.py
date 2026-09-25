from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _observed_ball(frame_id: int = 120) -> BallState:
    return BallState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 60.0,
        status=BallStateStatus.OBSERVED,
        x_px=20.0,
        y_px=20.0,
        confidence=0.9,
        uncertainty=0.1,
        source="test",
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


def test_v6_b_eight_direct_rim_observations_are_enough_for_miss() -> None:
    tracker = ShotEventTrackerV13(fps=60.0)
    event = _shot_event(
        semantic_start=False,
        had_above_rim=True,
        had_rim_band=True,
        direct_ball_observations=8,
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


def test_v6_b_rearm_requires_post_shot_control_reacquisition() -> None:
    tracker = ShotEventTrackerV13(fps=60.0)
    tracker._awaiting_rearm = True
    tracker._last_resolved_frame = 100
    tracker._rearm_clear_frames = tracker.rearm_clear_observed_frames
    relative = {
        "above": False,
        "near_x": False,
        "rim_band": False,
        "below": False,
        "inside_mouth": False,
    }
    assert tracker._update_rearm(
        frame_id=120,
        ball_state=_observed_ball(120),
        relative=relative,
        semantic_start=False,
        trajectory_start=False,
    ) is False
    assert tracker._awaiting_rearm is True

    tracker.confirm_control_reacquisition(119)
    assert tracker._update_rearm(
        frame_id=121,
        ball_state=_observed_ball(121),
        relative=relative,
        semantic_start=False,
        trajectory_start=False,
    ) is True
    assert tracker._awaiting_rearm is False


def test_v6_c_exact_actor_snapshot_can_publish_team() -> None:
    engine = P2EngineV6Batch(fps=60.0)
    engine._v6_team_evidence = {416: ("team_b", 0.97)}
    engine._v6_team_hints = {}
    team, confidence, age, source = engine._exact_actor_team_snapshot(416, frame_id=100)
    assert team == "team_b"
    assert confidence == 0.97
    assert age == 0
    assert source == "exact_actor_current_player_team"


def test_v6_c_actor_team_snapshot_is_frozen_at_release() -> None:
    engine = P2EngineV6Batch(fps=60.0)
    engine._pre_release_team_snapshot_source = "exact_actor_current_player_team"
    engine._pre_release_team_snapshot_confidence = 0.97
    engine._pre_release_team_snapshot_age_frames = 0
    engine._freeze_pre_release_snapshot_for_active_shot()

    engine._pre_release_team_snapshot_source = "unknown"
    engine._pre_release_team_snapshot_confidence = 0.0
    engine._pre_release_team_snapshot_age_frames = None
    source, confidence, age = engine._shot_snapshot_for_publication()
    assert source == "exact_actor_current_player_team"
    assert confidence == 0.97
    assert age == 0


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
    test_v6_b_eight_direct_rim_observations_are_enough_for_miss()
    test_v6_b_supported_flight_can_bridge_at_rim_band()
    test_v6_b_rearm_requires_post_shot_control_reacquisition()
    test_v6_c_exact_actor_snapshot_can_publish_team()
    test_v6_c_actor_team_snapshot_is_frozen_at_release()
    test_v6_d_limited_empty_scope_bootstrap_is_bounded()
    print("p2_v6_abcd_batch_test: PASS")


if __name__ == "__main__":
    main()
