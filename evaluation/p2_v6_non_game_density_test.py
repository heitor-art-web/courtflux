from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.court_state_gameplay_v5 import CourtStateEstimatorV5


def _step(
    estimator: CourtStateEstimatorV5,
    *,
    reason: str,
    fps: float = 10.0,
):
    return estimator._resolve_temporal_presence(
        candidate_present=False,
        candidate_status="non_gameplay",
        candidate_reason=reason,
        fps=fps,
        strong_live_evidence=False,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )


def test_v6_v5_contract_constants() -> None:
    estimator = CourtStateEstimatorV5()
    assert estimator.STATIC_FORMATION_CONFIRM_SEC == 2.2
    assert estimator.STATIC_HOLD_SEC == 4.0
    assert estimator.STALLED_CONTEXT_BALL_RATE_MAX == 0.65
    assert estimator.STALLED_WIDE_CONFIRM_SEC == 1.0
    assert estimator.STALLED_WIDE_HOLD_SEC == 2.5
    assert estimator.POLICY_VERSION == "court_state_gameplay_v5_v6_stoppage_density"


def test_v6_v5_static_hold_is_bounded_and_fail_open() -> None:
    estimator = CourtStateEstimatorV5()
    result = _step(estimator, reason="v5_static_formation_candidate")
    assert result[0] is False
    assert estimator._v5_hold_source == "static_formation"
    assert estimator._v5_hold_limit_frames == 40

    # Frames 2..39 remain NON_GAMEPLAY.
    for _ in range(38):
        result = _step(estimator, reason="ordinary_negative_candidate")
        assert result[0] is False

    # Frame 40 exhausts the bounded evidence budget and fails open.
    result = _step(estimator, reason="ordinary_negative_candidate")
    assert result[0] is True
    assert result[1] == "gameplay_limited"
    assert result[2] == "v5_static_formation_bounded_hold_expired_fail_open"
    assert estimator._v5_hold_source is None
    assert estimator._v5_static_source_blocked is True


def test_v6_v5_stalled_hold_is_shorter_and_bounded() -> None:
    estimator = CourtStateEstimatorV5()
    result = _step(estimator, reason="v5_stalled_wide_candidate")
    assert result[0] is False
    assert estimator._v5_hold_source == "stalled_wide"
    assert estimator._v5_hold_limit_frames == 25

    for _ in range(23):
        result = _step(estimator, reason="ordinary_negative_candidate")
        assert result[0] is False

    result = _step(estimator, reason="ordinary_negative_candidate")
    assert result[0] is True
    assert result[1] == "gameplay_limited"
    assert result[2] == "v5_stalled_wide_bounded_hold_expired_fail_open"
    assert estimator._v5_stalled_source_blocked is True


def test_v6_v5_reset_clears_bounded_hold_state() -> None:
    estimator = CourtStateEstimatorV5()
    _step(estimator, reason="v5_static_formation_candidate")
    assert estimator._v5_hold_source == "static_formation"
    estimator._reset_temporal_presence()
    assert estimator._v5_hold_source is None
    assert estimator._v5_hold_frames == 0
    assert estimator._v5_static_source_blocked is False
    assert estimator._v5_stalled_source_blocked is False


def main() -> None:
    test_v6_v5_contract_constants()
    test_v6_v5_static_hold_is_bounded_and_fail_open()
    test_v6_v5_stalled_hold_is_shorter_and_bounded()
    test_v6_v5_reset_clears_bounded_hold_state()
    print("p2_v6_non_game_density_test: PASS")


if __name__ == "__main__":
    main()
