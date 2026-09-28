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


def _frame(*, direct_ball: bool = True, surfaces: list[float] | None = None):
    values = surfaces or [0.42, 0.38, 0.34, 0.33, 0.31, 0.30, 0.29, 0.28, 0.35]
    return SimpleNamespace(
        players=[_player(value) for value in values],
        ball={"interpolated": not direct_ball},
        diagnostics={"detector": {"selected_ball": direct_ball}},
        fps=60.0,
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


def _start_static_hold(estimator: CourtStateEstimatorV6Resume, *, fps: float = 60.0):
    result = estimator._resolve_temporal_presence(
        candidate_present=False,
        candidate_status="non_gameplay",
        candidate_reason="v5_static_formation_candidate",
        fps=fps,
        strong_live_evidence=False,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    assert result[0] is False
    assert estimator._v5_hold_source == "static_formation"
    return result


def _resume_step(
    estimator: CourtStateEstimatorV6Resume,
    *,
    frame,
    state,
    evidence,
    fps: float = 60.0,
):
    confirmed, diagnostics = estimator._update_wide_dense_ball_resume_signal(
        frame,
        state=state,
        evidence=evidence,
        fps=fps,
    )
    result = estimator._resolve_temporal_presence(
        candidate_present=confirmed,
        candidate_status="gameplay_analysis" if confirmed else "non_gameplay",
        candidate_reason=(
            "v6_wide_dense_ball_resume_evidence"
            if confirmed
            else "ordinary_negative_candidate"
        ),
        fps=fps,
        strong_live_evidence=confirmed,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )
    return confirmed, diagnostics, result


def test_v6_resume_contract_constants() -> None:
    estimator = CourtStateEstimatorV6Resume()
    assert estimator.WIDE_DENSE_RESUME_MIN_PLAYERS == 8
    assert estimator.WIDE_DENSE_RESUME_MIN_ON_COURT == 5
    assert estimator.WIDE_DENSE_RESUME_MAX_HEIGHT_RATIO == 0.24
    assert estimator.WIDE_DENSE_RESUME_MIN_SURFACE_MEAN == 0.30
    assert estimator.WIDE_DENSE_RESUME_BASE_CONFIRM_SEC == 0.70
    assert estimator.STATIC_HOLD_SEC == 5.0
    assert estimator.STALLED_WIDE_HOLD_SEC == 3.0
    assert estimator.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC == 0.30
    assert (
        estimator.POLICY_VERSION
        == "court_state_gameplay_v6_v6_wide_dense_ball_resume"
    )


def test_v6_resume_base_accepts_direct_or_predicted_ball_only_during_static_hold() -> None:
    estimator = CourtStateEstimatorV6Resume()
    state = _state()
    evidence = _evidence()

    for direct_ball in (True, False):
        signal, diagnostics = estimator._wide_dense_ball_resume_base_signal(
            _frame(direct_ball=direct_ball),
            state=state,
            evidence=evidence,
        )
        assert signal is False
        assert diagnostics["v6_wide_dense_resume_hold_source"] is None

    estimator._v5_hold_source = "static_formation"
    for direct_ball in (True, False):
        signal, diagnostics = estimator._wide_dense_ball_resume_base_signal(
            _frame(direct_ball=direct_ball),
            state=state,
            evidence=evidence,
        )
        assert signal is True
        assert diagnostics["v6_wide_dense_resume_ball_present"] is True

    estimator._v5_hold_source = "stalled_wide"
    signal, _ = estimator._wide_dense_ball_resume_base_signal(
        _frame(direct_ball=True),
        state=state,
        evidence=evidence,
    )
    assert signal is False


def test_v6_resume_base_requires_wide_dense_court_supported_context() -> None:
    estimator = CourtStateEstimatorV6Resume()
    estimator._v5_hold_source = "static_formation"

    cases = [
        (_frame(), _state(players=7), _evidence()),
        (_frame(), _state(on_court=4), _evidence()),
        (_frame(), _state(), _evidence(height=0.25)),
        (_frame(surfaces=[0.10] * 9), _state(), _evidence()),
        (_frame(), _state(), _evidence(ball_present=False)),
    ]
    for frame, state, evidence in cases:
        signal, _ = estimator._wide_dense_ball_resume_base_signal(
            frame,
            state=state,
            evidence=evidence,
        )
        assert signal is False


def test_v6_resume_real_fps_requires_about_one_second_total_context() -> None:
    estimator = CourtStateEstimatorV6Resume()
    _start_static_hold(estimator, fps=60.0)
    frame = _frame(direct_ball=True)
    state = _state()
    evidence = _evidence()

    # At 60 fps, base confirmation is 42 frames. The existing V6 strong-live release
    # still needs 18 confirmed frames, so the hold must not release before ~1 second.
    for index in range(58):
        confirmed, diagnostics, result = _resume_step(
            estimator,
            frame=frame,
            state=state,
            evidence=evidence,
            fps=60.0,
        )
        assert result[0] is False
        if index < 41:
            assert confirmed is False
        else:
            assert confirmed is True
    assert diagnostics["v6_wide_dense_resume_confirm_frames"] == 42

    confirmed, _, result = _resume_step(
        estimator,
        frame=frame,
        state=state,
        evidence=evidence,
        fps=60.0,
    )
    assert confirmed is True
    assert result[0] is True
    assert result[2] == "v6_static_formation_hold_early_release_strong_live"
    assert estimator._v5_hold_source is None


def test_v6_resume_ng2_length_base_streak_does_not_release() -> None:
    estimator = CourtStateEstimatorV6Resume()
    _start_static_hold(estimator, fps=60.0)
    frame = _frame(direct_ball=True)
    state = _state()
    evidence = _evidence()

    # Frozen V6 NG2's closest wide+dense+ball streak is about 0.70 s. Even if the base
    # confirmation is reached on its final frame, there is no remaining 0.30 s strong-live
    # budget, so the hold must stay closed.
    for _ in range(42):
        _, _, result = _resume_step(
            estimator,
            frame=frame,
            state=state,
            evidence=evidence,
            fps=60.0,
        )
        assert result[0] is False

    # Clearing the base context immediately resets the V6 strong-live persistence.
    cleared_evidence = _evidence(ball_present=False)
    confirmed, _, result = _resume_step(
        estimator,
        frame=frame,
        state=state,
        evidence=cleared_evidence,
        fps=60.0,
    )
    assert confirmed is False
    assert result[0] is False
    assert estimator._v6_hold_strong_live_frames == 0


def main() -> None:
    test_v6_resume_contract_constants()
    test_v6_resume_base_accepts_direct_or_predicted_ball_only_during_static_hold()
    test_v6_resume_base_requires_wide_dense_court_supported_context()
    test_v6_resume_real_fps_requires_about_one_second_total_context()
    test_v6_resume_ng2_length_base_streak_does_not_release()
    print("p2_v6_resume_evidence_test: PASS")


if __name__ == "__main__":
    main()
