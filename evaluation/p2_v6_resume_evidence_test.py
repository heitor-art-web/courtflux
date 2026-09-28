from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.court_state_gameplay_v6_resume import CourtStateEstimatorV6Resume


def _player(surface: float) -> dict[str, object]:
    return {
        "local_court_surface_ratio": surface,
        "active_on_court": True,
    }


def _frame(*, direct_ball: bool, surfaces: list[float] | None = None):
    values = surfaces or [0.42, 0.38, 0.34, 0.33, 0.31, 0.30, 0.29, 0.28, 0.35]
    return SimpleNamespace(
        players=[_player(value) for value in values],
        ball={"interpolated": not direct_ball},
        diagnostics={"detector": {"selected_ball": direct_ball}},
    )


def _state(*, players: int = 9, on_court: int = 6):
    return SimpleNamespace(
        player_count=players,
        on_court_player_count=on_court,
        analysis_eligible=True,
    )


def _evidence(*, height: float = 0.20, ball_present: bool = True):
    return {
        "player_height_ratio_median": height,
        "ball_present": ball_present,
        "strong_closeup": False,
        "prefilter_scene_cut": False,
    }


def test_v6_resume_contract_constants() -> None:
    estimator = CourtStateEstimatorV6Resume()
    assert estimator.WIDE_PREDICTED_RESUME_MIN_PLAYERS == 8
    assert estimator.WIDE_PREDICTED_RESUME_MIN_ON_COURT == 5
    assert estimator.WIDE_PREDICTED_RESUME_MAX_HEIGHT_RATIO == 0.24
    assert estimator.WIDE_PREDICTED_RESUME_MIN_SURFACE_MEAN == 0.30
    assert estimator.STATIC_HOLD_SEC == 5.0
    assert estimator.STALLED_WIDE_HOLD_SEC == 3.0
    assert estimator.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC == 0.30
    assert (
        estimator.POLICY_VERSION
        == "court_state_gameplay_v6_v6_wide_predicted_ball_resume"
    )


def test_v6_resume_signal_accepts_wide_predicted_ball_only_during_hold() -> None:
    estimator = CourtStateEstimatorV6Resume()
    frame = _frame(direct_ball=False)
    state = _state()
    evidence = _evidence()

    signal, _ = estimator._wide_predicted_ball_resume_signal(
        frame,
        state=state,
        evidence=evidence,
    )
    assert signal is False

    estimator._v5_hold_source = "static_formation"
    signal, diagnostics = estimator._wide_predicted_ball_resume_signal(
        frame,
        state=state,
        evidence=evidence,
    )
    assert signal is True
    assert diagnostics["v6_wide_predicted_resume_direct_ball"] is False
    assert diagnostics["v6_wide_predicted_resume_surface_mean"] >= 0.30


def test_v6_resume_signal_rejects_direct_ball_ng2_signature() -> None:
    estimator = CourtStateEstimatorV6Resume()
    estimator._v5_hold_source = "static_formation"
    signal, diagnostics = estimator._wide_predicted_ball_resume_signal(
        _frame(direct_ball=True),
        state=_state(),
        evidence=_evidence(),
    )
    assert signal is False
    assert diagnostics["v6_wide_predicted_resume_direct_ball"] is True


def test_v6_resume_signal_requires_wide_dense_court_supported_context() -> None:
    estimator = CourtStateEstimatorV6Resume()
    estimator._v5_hold_source = "static_formation"

    signal, _ = estimator._wide_predicted_ball_resume_signal(
        _frame(direct_ball=False),
        state=_state(players=7),
        evidence=_evidence(),
    )
    assert signal is False

    signal, _ = estimator._wide_predicted_ball_resume_signal(
        _frame(direct_ball=False),
        state=_state(on_court=4),
        evidence=_evidence(),
    )
    assert signal is False

    signal, _ = estimator._wide_predicted_ball_resume_signal(
        _frame(direct_ball=False),
        state=_state(),
        evidence=_evidence(height=0.25),
    )
    assert signal is False

    signal, _ = estimator._wide_predicted_ball_resume_signal(
        _frame(direct_ball=False, surfaces=[0.10] * 9),
        state=_state(),
        evidence=_evidence(),
    )
    assert signal is False


def test_v6_resume_signal_still_uses_existing_sustained_release_horizon() -> None:
    estimator = CourtStateEstimatorV6Resume()

    result = estimator._resolve_temporal_presence(
        candidate_present=False,
        candidate_status="non_gameplay",
        candidate_reason="v5_static_formation_candidate",
        fps=10.0,
        strong_live_evidence=False,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    assert result[0] is False
    assert estimator._v5_hold_source == "static_formation"

    for _ in range(2):
        result = estimator._resolve_temporal_presence(
            candidate_present=True,
            candidate_status="gameplay_analysis",
            candidate_reason="v6_wide_predicted_ball_resume_evidence",
            fps=10.0,
            strong_live_evidence=True,
            safe_limited_resume=False,
            weak_live_evidence=False,
        )
        assert result[0] is False

    result = estimator._resolve_temporal_presence(
        candidate_present=True,
        candidate_status="gameplay_analysis",
        candidate_reason="v6_wide_predicted_ball_resume_evidence",
        fps=10.0,
        strong_live_evidence=True,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    assert result[0] is True
    assert result[2] == "v6_static_formation_hold_early_release_strong_live"


def main() -> None:
    test_v6_resume_contract_constants()
    test_v6_resume_signal_accepts_wide_predicted_ball_only_during_hold()
    test_v6_resume_signal_rejects_direct_ball_ng2_signature()
    test_v6_resume_signal_requires_wide_dense_court_supported_context()
    test_v6_resume_signal_still_uses_existing_sustained_release_horizon()
    print("p2_v6_resume_evidence_test: PASS")


if __name__ == "__main__":
    main()
