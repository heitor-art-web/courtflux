from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.court_state import CourtState
from services.vision.court_state_gameplay_v2 import CourtStateEstimatorV2


def _state(
    *,
    analysis_eligible: bool = False,
    player_count: int = 0,
    on_court_player_count: int = 0,
    basket: bool = False,
) -> CourtState:
    return CourtState(
        frame_id=1,
        timestamp_sec=0.1,
        analysis_eligible=analysis_eligible,
        player_count=player_count,
        on_court_player_count=on_court_player_count,
        basket_evidence=([{"kind": "rim"}] if basket else []),
        gameplay_present=True,
    )


def test_sustained_negative_evidence_enters_non_gameplay_without_single_frame_flicker() -> None:
    estimator = CourtStateEstimatorV2(
        non_gameplay_enter_sec=0.60,
        gameplay_resume_sec=0.20,
    )

    # Establish gameplay first.
    present, status, _, _ = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_analysis",
        candidate_reason="test_live",
        fps=10.0,
    )
    assert present is True and status == "gameplay_analysis"

    # 0.5 s of negative evidence is still debounce/continuity, not a hard boundary.
    for _ in range(5):
        present, status, _, _ = estimator._resolve_temporal_presence(
            candidate_present=False,
            candidate_status="non_gameplay",
            candidate_reason="test_dead",
            fps=10.0,
        )
        assert present is True
        assert status == "gameplay_limited"

    # The sixth 10-fps frame reaches 0.6 s and confirms NON_GAMEPLAY.
    present, status, reason, _ = estimator._resolve_temporal_presence(
        candidate_present=False,
        candidate_status="non_gameplay",
        candidate_reason="test_dead",
        fps=10.0,
    )
    assert present is False
    assert status == "non_gameplay"
    assert reason == "sustained_non_gameplay_confirmed"


def test_gameplay_resume_requires_short_positive_confirmation() -> None:
    estimator = CourtStateEstimatorV2(
        non_gameplay_enter_sec=0.20,
        gameplay_resume_sec=0.20,
    )
    # Confirm non-gameplay in two frames at 10 fps.
    for _ in range(2):
        result = estimator._resolve_temporal_presence(
            candidate_present=False,
            candidate_status="non_gameplay",
            candidate_reason="dead",
            fps=10.0,
        )
    assert result[0] is False

    first = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_analysis",
        candidate_reason="live",
        fps=10.0,
    )
    assert first[0] is False
    assert first[2] == "gameplay_resume_pending"

    second = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_analysis",
        candidate_reason="live",
        fps=10.0,
    )
    assert second[0] is True
    assert second[1] == "gameplay_analysis"


def test_free_throw_like_basket_ball_context_stays_limited_not_non_gameplay() -> None:
    estimator = CourtStateEstimatorV2(limited_motion_ratio=0.006)
    state = _state(
        analysis_eligible=False,
        player_count=6,
        on_court_player_count=2,
        basket=True,
    )
    frame = SimpleNamespace(ball={"box": [1, 1, 2, 2]})

    candidate = estimator._candidate(
        frame,
        state,
        {
            "likely_gameplay": False,
            "scene_cut": False,
            "motion_ratio": 0.001,
        },
    )
    assert candidate is not None
    assert candidate[0] is True
    assert candidate[1] == "gameplay_limited"
    assert candidate[2] == "basket_ball_context_hold"


def test_explicit_on_court_motion_preserves_limited_gameplay_when_prefilter_abstains() -> None:
    estimator = CourtStateEstimatorV2(limited_motion_ratio=0.006)
    state = _state(
        analysis_eligible=False,
        player_count=5,
        on_court_player_count=3,
        basket=False,
    )
    frame = SimpleNamespace(ball=None)

    candidate = estimator._candidate(
        frame,
        state,
        {
            "likely_gameplay": False,
            "scene_cut": False,
            "motion_ratio": 0.012,
        },
    )
    assert candidate is not None
    assert candidate[:3] == (
        True,
        "gameplay_limited",
        "on_court_motion_hold",
    )


def test_scene_cut_is_negative_candidate_but_temporal_gate_prevents_one_frame_boundary() -> None:
    estimator = CourtStateEstimatorV2(
        non_gameplay_enter_sec=0.50,
        gameplay_resume_sec=0.20,
    )
    estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_analysis",
        candidate_reason="live",
        fps=10.0,
    )

    candidate = estimator._candidate(
        SimpleNamespace(ball=None),
        _state(analysis_eligible=True, player_count=7, on_court_player_count=4),
        {
            "likely_gameplay": False,
            "scene_cut": True,
            "motion_ratio": 0.20,
        },
    )
    assert candidate is not None and candidate[0] is False
    present, status, _, _ = estimator._resolve_temporal_presence(
        candidate_present=candidate[0],
        candidate_status=candidate[1],
        candidate_reason=candidate[2],
        fps=10.0,
    )
    assert present is True
    assert status == "gameplay_limited"


def test_missing_prefilter_diagnostics_preserves_v1_policy() -> None:
    estimator = CourtStateEstimatorV2()
    candidate = estimator._candidate(
        SimpleNamespace(ball=None),
        _state(analysis_eligible=False, player_count=3),
        {},
    )
    assert candidate is None


def main() -> None:
    test_sustained_negative_evidence_enters_non_gameplay_without_single_frame_flicker()
    test_gameplay_resume_requires_short_positive_confirmation()
    test_free_throw_like_basket_ball_context_stays_limited_not_non_gameplay()
    test_explicit_on_court_motion_preserves_limited_gameplay_when_prefilter_abstains()
    test_scene_cut_is_negative_candidate_but_temporal_gate_prevents_one_frame_boundary()
    test_missing_prefilter_diagnostics_preserves_v1_policy()
    print("court_state_gameplay_v2_test: PASS")


if __name__ == "__main__":
    main()
