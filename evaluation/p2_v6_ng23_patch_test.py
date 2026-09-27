from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.court_state_gameplay_v4 import CourtStateEstimatorV4


def _static_evidence(*, motion: float = 0.006) -> dict:
    return {
        "v3_action_window_motion_mean": motion,
        "v3_action_window_valid_motion_ratio": 1.0,
        "player_height_ratio_median": 0.24,
        "strong_closeup": False,
        "prefilter_scene_cut": False,
    }


def test_v4_contract() -> None:
    estimator = CourtStateEstimatorV4()
    assert estimator.non_gameplay_enter_sec == 4.5
    assert estimator.STATIC_FORMATION_CONFIRM_SEC == 4.0
    assert estimator.STALLED_CONTEXT_WINDOW_SEC == 2.0
    assert estimator.STALLED_CONTEXT_ANALYSIS_RATE_MAX == 0.15
    assert estimator.STALLED_CONTEXT_BALL_RATE_MAX == 0.35


def test_static_formation_requires_four_seconds() -> None:
    estimator = CourtStateEstimatorV4()
    state = SimpleNamespace(player_count=8, on_court_player_count=5)
    result = None
    for _ in range(39):
        result = estimator._update_static_formation(
            state=state,
            evidence=_static_evidence(),
            fps=10.0,
        )
        assert result["v4_static_formation_confirmed"] is False

    result = estimator._update_static_formation(
        state=state,
        evidence=_static_evidence(),
        fps=10.0,
    )
    assert result["v4_static_formation_confirmed"] is True

    resolved = estimator._resolve_temporal_presence(
        candidate_present=False,
        candidate_status="non_gameplay",
        candidate_reason="v4_static_formation_candidate",
        fps=10.0,
        strong_live_evidence=False,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    assert resolved[0] is False
    assert resolved[1] == "non_gameplay"
    assert resolved[2] == "v4_static_formation_confirmed"


def test_static_formation_resets_on_real_relative_motion() -> None:
    estimator = CourtStateEstimatorV4()
    state = SimpleNamespace(player_count=8, on_court_player_count=5)
    for _ in range(30):
        estimator._update_static_formation(
            state=state,
            evidence=_static_evidence(),
            fps=10.0,
        )
    result = estimator._update_static_formation(
        state=state,
        evidence=_static_evidence(motion=0.02),
        fps=10.0,
    )
    assert result["v4_static_formation_signal"] is False
    assert result["v4_static_formation_frames"] == 0
    assert result["v4_static_formation_confirmed"] is False


def test_stalled_context_requires_full_two_second_history() -> None:
    estimator = CourtStateEstimatorV4()
    for _ in range(19):
        estimator._v4_stalled_context_history.append(
            {
                "analysis_eligible": False,
                "ball_present": False,
                "on_court_count": 4,
                "surface_ratio_median": 0.15,
            }
        )
    assert estimator._stalled_context_summary(fps=10.0)[
        "v4_stalled_context_ready"
    ] is False

    estimator._v4_stalled_context_history.append(
        {
            "analysis_eligible": False,
            "ball_present": False,
            "on_court_count": 4,
            "surface_ratio_median": 0.15,
        }
    )
    summary = estimator._stalled_context_summary(fps=10.0)
    assert summary["v4_stalled_context_ready"] is True
    assert summary["v4_stalled_context_analysis_rate"] == 0.0
    assert summary["v4_stalled_context_ball_rate"] == 0.0


def test_stalled_context_rejects_active_or_ball_rich_history() -> None:
    estimator = CourtStateEstimatorV4()
    for index in range(20):
        estimator._v4_stalled_context_history.append(
            {
                "analysis_eligible": index < 10,
                "ball_present": index < 10,
                "on_court_count": 5,
                "surface_ratio_median": 0.15,
            }
        )
    summary = estimator._stalled_context_summary(fps=10.0)
    assert summary["v4_stalled_context_ready"] is False
    assert summary["v4_stalled_context_analysis_rate"] == 0.5
    assert summary["v4_stalled_context_ball_rate"] == 0.5


def test_stalled_context_broadcast_break_direct_confirmation() -> None:
    estimator = CourtStateEstimatorV4()
    resolved = estimator._resolve_temporal_presence(
        candidate_present=False,
        candidate_status="non_gameplay",
        candidate_reason="v4_stalled_context_broadcast_break_candidate",
        fps=10.0,
        strong_live_evidence=False,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    assert resolved[0] is False
    assert resolved[1] == "non_gameplay"
    assert resolved[2] == "v4_stalled_context_broadcast_break_confirmed"
    assert resolved[3]["v4_direct_confirmation"] is True


def test_surface_ratio_median_ignores_missing_values() -> None:
    frame = SimpleNamespace(
        players=[
            {"local_court_surface_ratio": 0.10},
            {"local_court_surface_ratio": None},
            {"local_court_surface_ratio": 0.30},
        ]
    )
    assert CourtStateEstimatorV4._surface_ratio_median(frame) == 0.20


def main() -> None:
    test_v4_contract()
    test_static_formation_requires_four_seconds()
    test_static_formation_resets_on_real_relative_motion()
    test_stalled_context_requires_full_two_second_history()
    test_stalled_context_rejects_active_or_ball_rich_history()
    test_stalled_context_broadcast_break_direct_confirmation()
    test_surface_ratio_median_ignores_missing_values()
    print("p2_v6_ng23_patch_test: PASS")


if __name__ == "__main__":
    main()
