from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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


def _motion_players(frame_id: int, *, global_pan_px: float = 0.0) -> list[dict]:
    shift = float(frame_id) * float(global_pan_px)
    return [
        {"track_id": 1, "box": [100.0 + shift, 100.0, 150.0 + shift, 200.0]},
        {"track_id": 2, "box": [250.0 + shift, 100.0, 300.0 + shift, 200.0]},
        {"track_id": 3, "box": [400.0 + shift, 100.0, 450.0 + shift, 200.0]},
        {"track_id": 4, "box": [550.0 + shift, 100.0, 600.0 + shift, 200.0]},
    ]


def test_v6_a_hysteresis_contract() -> None:
    estimator = CourtStateEstimatorV3()
    assert estimator.non_gameplay_enter_sec == 3.5
    assert estimator.weak_live_hold_sec == 1.0
    assert estimator.gameplay_resume_sec == 0.20
    assert estimator.LOW_ACTION_CONFIRM_SEC == 4.0
    assert estimator.ACTION_WINDOW_SEC == 1.5


def test_v6_a_weak_live_timeout_persists_until_strong_resume() -> None:
    estimator = CourtStateEstimatorV3()

    def weak():
        return estimator._resolve_temporal_presence(
            candidate_present=True,
            candidate_status="gameplay_limited",
            candidate_reason="motion_supported_prefilter",
            fps=10.0,
            strong_live_evidence=False,
            safe_limited_resume=False,
            weak_live_evidence=True,
        )

    result = None
    for _ in range(9):
        result = weak()
    assert result is not None and result[0] is True
    assert estimator._v3_weak_timeout_latched is False

    result = weak()
    assert result[0] is True
    assert estimator._v3_weak_timeout_latched is True
    for _ in range(34):
        result = weak()
    assert result[0] is False
    assert result[1] == "non_gameplay"

    result = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_analysis",
        candidate_reason="strong_wide_live_evidence",
        fps=10.0,
        strong_live_evidence=True,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    assert result[0] is True
    assert estimator._v3_weak_timeout_latched is False


def test_v6_a_safe_basket_context_does_not_consume_weak_budget() -> None:
    estimator = CourtStateEstimatorV3()
    for _ in range(100):
        result = estimator._resolve_temporal_presence(
            candidate_present=True,
            candidate_status="gameplay_limited",
            candidate_reason="basket_ball_context_hold",
            fps=10.0,
            strong_live_evidence=False,
            safe_limited_resume=True,
            weak_live_evidence=True,
        )
        assert result[0] is True
    assert estimator._v3_weak_timeout_latched is False
    assert estimator._v3_weak_context_frames == 0


def test_v6_a_low_action_watchdog_ignores_global_camera_pan() -> None:
    estimator = CourtStateEstimatorV3()
    diagnostics = None
    # All player boxes translate together. Residual player motion is therefore near zero:
    # camera movement alone must not keep a dead-ball sequence classified as live action.
    for frame_id in range(70):
        frame = SimpleNamespace(
            players=_motion_players(frame_id, global_pan_px=3.0),
            ball=None,
            fps=10.0,
        )
        diagnostics = estimator._update_low_action_watchdog(frame, fps=10.0)
    assert diagnostics is not None
    assert diagnostics["v3_action_window_motion_mean"] == 0.0
    assert diagnostics["v3_action_window_direct_ball_rate"] == 0.0
    assert diagnostics["v3_low_action_confirmed"] is True

    result = estimator._resolve_temporal_presence(
        candidate_present=False,
        candidate_status="non_gameplay",
        candidate_reason="v3_sustained_low_action_candidate",
        fps=10.0,
        strong_live_evidence=False,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    assert result[0] is False
    assert result[1] == "non_gameplay"
    assert result[2] == "v3_sustained_low_action_confirmed"


def test_v6_b_ft_single_mouth_plus_descent_confirms_made() -> None:
    reason = ShotEventTrackerV13._deferred_confirmation_reason(
        {
            "rim_observations": 6,
            "near_x_observations": 1,
            "inside_mouth_observations": 1,
            "below_rim_observations": 6,
            "descending_steps": 4,
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


def test_v6_b_unresolved_supported_flight_bridge_is_not_published() -> None:
    tracker = ShotEventTrackerV13(fps=60.0)
    event = _shot_event(
        semantic_start=False,
        had_above_rim=False,
        had_rim_band=True,
        direct_ball_observations=16,
        resolution_reason="shot_window_expired_without_outcome_evidence",
        pending_bridge_used=False,
    )
    event.evidence = ("ball_near_rim", "supported_airborne_rim_band_bridge")
    assert tracker._postprocess_public_event(event) is None
    diagnostics = tracker.diagnostics()
    assert diagnostics["suppressed_unresolved_supported_bridges"] == 1
    assert diagnostics["last_suppressed_reason"] == (
        "supported_flight_bridge_terminal_outcome_unresolved"
    )


def test_v6_b_supported_bridge_with_strong_rim_evidence_survives_as_miss() -> None:
    tracker = ShotEventTrackerV13(fps=60.0)
    event = _shot_event(
        semantic_start=False,
        had_above_rim=True,
        had_rim_band=True,
        direct_ball_observations=10,
        resolution_reason="shot_window_expired_without_outcome_evidence",
        pending_bridge_used=False,
    )
    event.evidence = (
        "ball_above_rim",
        "ball_near_rim",
        "supported_airborne_rim_band_bridge",
    )
    result = tracker._postprocess_public_event(event)
    assert result is not None
    assert result.attributes["outcome"] == ShotOutcome.MISSED.value
    assert result.decision is EventDecision.CONFIRMED
    assert tracker.diagnostics()["suppressed_unresolved_supported_bridges"] == 0


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


def test_v6_b_nested_possession_flight_support_is_consumed() -> None:
    assert P2EngineV6Batch._supported_flight_from_possession_diagnostics(
        {"flight": {"flight_supported": True}}
    ) is True
    assert P2EngineV6Batch._supported_flight_from_possession_diagnostics(
        {"flight": {"flight_supported": False}, "flight_supported": True}
    ) is False


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


def test_v6_c_actorless_shot_uses_frozen_public_flight_origin() -> None:
    engine = P2EngineV6Batch(fps=60.0)
    engine.possession_tracker.diagnostics = lambda: {
        "flight": {
            "active": True,
            "origin_track_id": 360,
            "origin_team_id": "team_a",
            "flight_supported": True,
        }
    }
    actor, team, source = engine._flight_origin_public_attribution(frame_id=100)
    assert actor == 360
    assert team == "team_a"
    assert source == "pre_release_flight_origin_public_team"
    assert engine._pre_release_team_snapshot_source == "flight_origin_public_team"


def test_v6_c_public_possession_team_abstains_before_switch() -> None:
    engine = P2EngineV6Batch(fps=60.0)

    def observe(raw_team: str | None) -> str | None:
        frame = SimpleNamespace(
            diagnostics={
                "p2": {
                    "possession": {
                        "control_state": "possessed",
                        "current_team_id": raw_team,
                    }
                }
            }
        )
        engine._stabilize_public_possession_team(frame)
        return frame.diagnostics["p2"]["possession"]["current_team_id"]

    assert observe("team_a") is None
    assert observe("team_a") is None
    assert observe("team_a") == "team_a"

    for _ in range(18):
        assert observe("team_b") is None
    assert observe("team_a") == "team_a"

    result = None
    for _ in range(engine._public_team_switch_confirm_frames()):
        result = observe("team_b")
    assert result == "team_b"


def test_v6_d_limited_empty_scope_bootstrap_is_tiered() -> None:
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
            {
                "track_id": 4,
                "active_on_court": None,
                "identity_eligible": True,
                "confidence": 0.76,
                "local_court_surface_ratio": 0.10,
                "court_eligibility_source": "image_space_ambiguous_surface",
                "team": None,
                "team_confidence": None,
            },
        ],
        known_track_ids={4},
    )
    assert [row["track_id"] for row in selected] == [4, 1]
    assert selected[0]["gate_d_limited_bootstrap_tier"] == "known_resolved_track"
    assert selected[1]["gate_d_limited_bootstrap_tier"] == (
        "persistent_new_track_candidate"
    )
    assert all(row["active_on_court"] is True for row in selected)


def main() -> None:
    test_v6_a_hysteresis_contract()
    test_v6_a_weak_live_timeout_persists_until_strong_resume()
    test_v6_a_safe_basket_context_does_not_consume_weak_budget()
    test_v6_a_low_action_watchdog_ignores_global_camera_pan()
    test_v6_b_ft_single_mouth_plus_descent_confirms_made()
    test_v6_b_incoherent_deferred_geometry_still_abstains()
    test_v6_b_unconfirmed_trajectory_start_is_not_published()
    test_v6_b_unresolved_supported_flight_bridge_is_not_published()
    test_v6_b_supported_bridge_with_strong_rim_evidence_survives_as_miss()
    test_v6_b_eight_direct_rim_observations_are_enough_for_miss()
    test_v6_b_supported_flight_can_bridge_at_rim_band()
    test_v6_b_nested_possession_flight_support_is_consumed()
    test_v6_b_rearm_requires_post_shot_control_reacquisition()
    test_v6_c_exact_actor_snapshot_can_publish_team()
    test_v6_c_actor_team_snapshot_is_frozen_at_release()
    test_v6_c_actorless_shot_uses_frozen_public_flight_origin()
    test_v6_c_public_possession_team_abstains_before_switch()
    test_v6_d_limited_empty_scope_bootstrap_is_tiered()
    print("p2_v6_abcd_batch_test: PASS")


if __name__ == "__main__":
    main()
