from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.court_state import CourtState
from services.p2.pipeline import P2Pipeline
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


def _view_frame(*, height_ratios: list[float], ball: bool = False) -> SimpleNamespace:
    frame_h = 1000
    players = []
    for index, ratio in enumerate(height_ratios):
        height = float(ratio) * frame_h
        players.append(
            {
                "track_id": index + 1,
                "box": [float(index * 50), 100.0, float(index * 50 + 40), 100.0 + height],
            }
        )
    return SimpleNamespace(
        frame=SimpleNamespace(shape=(frame_h, 1800, 3)),
        players=players,
        ball=({"box": [10, 10, 20, 20]} if ball else None),
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
        strong_gameplay_resume_sec=0.20,
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
    assert candidate[3]["safe_limited_resume"] is True


def test_basket_ball_context_can_resume_limited_after_confirmed_stoppage() -> None:
    estimator = CourtStateEstimatorV2(
        non_gameplay_enter_sec=0.20,
        gameplay_resume_sec=0.20,
    )
    for _ in range(2):
        result = estimator._resolve_temporal_presence(
            candidate_present=False,
            candidate_status="non_gameplay",
            candidate_reason="dead",
            fps=10.0,
            strong_live_evidence=False,
        )
    assert result[0] is False

    first = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_limited",
        candidate_reason="basket_ball_context_hold",
        fps=10.0,
        strong_live_evidence=False,
        safe_limited_resume=True,
        weak_live_evidence=True,
    )
    second = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_limited",
        candidate_reason="basket_ball_context_hold",
        fps=10.0,
        strong_live_evidence=False,
        safe_limited_resume=True,
        weak_live_evidence=True,
    )
    assert first[0] is False
    assert second[0] is True
    assert second[1] == "gameplay_limited"


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
    assert candidate[3]["weak_live_evidence"] is True


def test_strong_closeup_overrides_motion_and_basket_context() -> None:
    estimator = CourtStateEstimatorV2(
        limited_motion_ratio=0.006,
        strong_closeup_height_ratio=0.55,
    )
    frame = _view_frame(height_ratios=[0.82, 0.88], ball=True)
    state = _state(
        analysis_eligible=False,
        player_count=2,
        on_court_player_count=2,
        basket=True,
    )
    candidate = estimator._candidate(
        frame,
        state,
        {
            "likely_gameplay": True,
            "scene_cut": False,
            "motion_ratio": 0.20,
        },
    )
    assert candidate is not None
    assert candidate[:3] == (False, "non_gameplay", "strong_closeup_candidate")
    assert candidate[3]["strong_closeup"] is True


def test_strong_wide_live_view_is_fast_resume_evidence() -> None:
    estimator = CourtStateEstimatorV2(
        wide_resume_min_players=6,
        wide_resume_max_height_ratio=0.35,
        wide_resume_motion_ratio=0.012,
    )
    frame = _view_frame(
        height_ratios=[0.18, 0.20, 0.21, 0.22, 0.24, 0.25, 0.27, 0.29],
        ball=True,
    )
    state = _state(
        analysis_eligible=False,
        player_count=8,
        on_court_player_count=4,
        basket=False,
    )
    candidate = estimator._candidate(
        frame,
        state,
        {
            "likely_gameplay": False,
            "scene_cut": False,
            "motion_ratio": 0.025,
        },
    )
    assert candidate is not None
    assert candidate[:3] == (True, "gameplay_limited", "strong_wide_live_evidence")
    assert candidate[3]["strong_live_evidence"] is True


def test_weak_live_context_is_bounded_and_eventually_enters_non_gameplay() -> None:
    estimator = CourtStateEstimatorV2(
        non_gameplay_enter_sec=0.20,
        weak_live_hold_sec=1.0,
    )
    estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_analysis",
        candidate_reason="strong_live",
        fps=10.0,
        strong_live_evidence=True,
    )

    for _ in range(9):
        present, _, _, _ = estimator._resolve_temporal_presence(
            candidate_present=True,
            candidate_status="gameplay_limited",
            candidate_reason="on_court_motion_hold",
            fps=10.0,
            strong_live_evidence=False,
            weak_live_evidence=True,
        )
        assert present is True

    # Frame 10 exhausts the weak hold and becomes the first negative debounce frame.
    tenth = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_limited",
        candidate_reason="on_court_motion_hold",
        fps=10.0,
        strong_live_evidence=False,
        weak_live_evidence=True,
    )
    assert tenth[0] is True
    assert tenth[1] == "gameplay_limited"

    eleventh = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_limited",
        candidate_reason="on_court_motion_hold",
        fps=10.0,
        strong_live_evidence=False,
        weak_live_evidence=True,
    )
    assert eleventh[0] is False
    assert eleventh[1] == "non_gameplay"
    assert eleventh[2] == "sustained_non_gameplay_confirmed"


def test_weak_player_motion_alone_does_not_reopen_confirmed_non_gameplay() -> None:
    estimator = CourtStateEstimatorV2(non_gameplay_enter_sec=0.20)
    for _ in range(2):
        result = estimator._resolve_temporal_presence(
            candidate_present=False,
            candidate_status="non_gameplay",
            candidate_reason="dead",
            fps=10.0,
            strong_live_evidence=False,
        )
    assert result[0] is False

    held = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_limited",
        candidate_reason="on_court_motion_hold",
        fps=10.0,
        strong_live_evidence=False,
        safe_limited_resume=False,
        weak_live_evidence=True,
    )
    assert held[0] is False
    assert held[2] == "gameplay_resume_waiting_strong_live_evidence"


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


def test_p2_pipeline_shares_stateful_gameplay_estimator_with_identity_gate() -> None:
    detector = SimpleNamespace(identity_court_estimator=object())
    pipeline = P2Pipeline(
        gameplay_filter=SimpleNamespace(),
        camera_tracker=None,
        detector=detector,
        database=None,
        renderer=None,
        profiler=SimpleNamespace(),
        backboard_roi=None,
        persist_results=False,
    )

    assert isinstance(pipeline.court_state_estimator, CourtStateEstimatorV2)
    assert detector.identity_court_estimator is pipeline.court_state_estimator


def main() -> None:
    test_sustained_negative_evidence_enters_non_gameplay_without_single_frame_flicker()
    test_gameplay_resume_requires_short_positive_confirmation()
    test_free_throw_like_basket_ball_context_stays_limited_not_non_gameplay()
    test_basket_ball_context_can_resume_limited_after_confirmed_stoppage()
    test_explicit_on_court_motion_preserves_limited_gameplay_when_prefilter_abstains()
    test_strong_closeup_overrides_motion_and_basket_context()
    test_strong_wide_live_view_is_fast_resume_evidence()
    test_weak_live_context_is_bounded_and_eventually_enters_non_gameplay()
    test_weak_player_motion_alone_does_not_reopen_confirmed_non_gameplay()
    test_scene_cut_is_negative_candidate_but_temporal_gate_prevents_one_frame_boundary()
    test_missing_prefilter_diagnostics_preserves_v1_policy()
    test_p2_pipeline_shares_stateful_gameplay_estimator_with_identity_gate()
    print("court_state_gameplay_v2_test: PASS")


if __name__ == "__main__":
    main()
