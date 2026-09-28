from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.court_state_gameplay_v6 import CourtStateEstimatorV6


def _step(
    estimator: CourtStateEstimatorV6,
    *,
    reason: str,
    present: bool,
    status: str,
    strong_live: bool,
    fps: float = 10.0,
):
    return estimator._resolve_temporal_presence(
        candidate_present=present,
        candidate_status=status,
        candidate_reason=reason,
        fps=fps,
        strong_live_evidence=strong_live,
        safe_limited_resume=False,
        weak_live_evidence=False,
    )


def test_v6_hold_release_contract_constants() -> None:
    estimator = CourtStateEstimatorV6()
    assert estimator.STATIC_HOLD_SEC == 5.0
    assert estimator.STALLED_WIDE_HOLD_SEC == 3.0
    assert estimator.HOLD_EARLY_RELEASE_STRONG_LIVE_SEC == 0.30
    assert estimator.POLICY_VERSION == "court_state_gameplay_v6_v6_hold_release"


def test_v6_static_hold_releases_on_sustained_strong_live() -> None:
    estimator = CourtStateEstimatorV6()

    result = _step(
        estimator,
        reason="v5_static_formation_candidate",
        present=False,
        status="non_gameplay",
        strong_live=False,
    )
    assert result[0] is False
    assert estimator._v5_hold_source == "static_formation"
    assert estimator._v5_hold_limit_frames == 50

    # A two-frame live blip is below the 0.30 s / 3-frame release horizon.
    for _ in range(2):
        result = _step(
            estimator,
            reason="strong_wide_live_evidence",
            present=True,
            status="gameplay_analysis",
            strong_live=True,
        )
        assert result[0] is False
        assert estimator._v5_hold_source == "static_formation"

    # The third consecutive strong-live frame releases immediately, long before max hold.
    result = _step(
        estimator,
        reason="strong_wide_live_evidence",
        present=True,
        status="gameplay_analysis",
        strong_live=True,
    )
    assert result[0] is True
    assert result[1] == "gameplay_analysis"
    assert result[2] == "v6_static_formation_hold_early_release_strong_live"
    assert estimator._v5_hold_source is None
    assert estimator._v5_static_source_blocked is True
    assert estimator._v6_early_release_count == 1


def test_v6_live_release_requires_consecutive_evidence() -> None:
    estimator = CourtStateEstimatorV6()
    _step(
        estimator,
        reason="v5_static_formation_candidate",
        present=False,
        status="non_gameplay",
        strong_live=False,
    )

    for _ in range(2):
        result = _step(
            estimator,
            reason="strong_wide_live_evidence",
            present=True,
            status="gameplay_analysis",
            strong_live=True,
        )
        assert result[0] is False

    # One negative frame resets the persistence counter.
    result = _step(
        estimator,
        reason="ordinary_negative_candidate",
        present=False,
        status="non_gameplay",
        strong_live=False,
    )
    assert result[0] is False
    assert estimator._v6_hold_strong_live_frames == 0

    for _ in range(2):
        result = _step(
            estimator,
            reason="strong_wide_live_evidence",
            present=True,
            status="gameplay_analysis",
            strong_live=True,
        )
        assert result[0] is False

    result = _step(
        estimator,
        reason="strong_wide_live_evidence",
        present=True,
        status="gameplay_analysis",
        strong_live=True,
    )
    assert result[0] is True


def test_v6_stalled_hold_keeps_longer_non_game_budget() -> None:
    estimator = CourtStateEstimatorV6()

    result = _step(
        estimator,
        reason="v5_stalled_wide_candidate",
        present=False,
        status="non_gameplay",
        strong_live=False,
    )
    assert result[0] is False
    assert estimator._v5_hold_source == "stalled_wide"
    assert estimator._v5_hold_limit_frames == 30

    # With no strong live evidence, the longer stoppage budget remains active.
    for _ in range(28):
        result = _step(
            estimator,
            reason="ordinary_negative_candidate",
            present=False,
            status="non_gameplay",
            strong_live=False,
        )
        assert result[0] is False

    result = _step(
        estimator,
        reason="ordinary_negative_candidate",
        present=False,
        status="non_gameplay",
        strong_live=False,
    )
    assert result[0] is True
    assert result[1] == "gameplay_limited"
    assert result[2] == "v5_stalled_wide_bounded_hold_expired_fail_open"


def main() -> None:
    test_v6_hold_release_contract_constants()
    test_v6_static_hold_releases_on_sustained_strong_live()
    test_v6_live_release_requires_consecutive_evidence()
    test_v6_stalled_hold_keeps_longer_non_game_budget()
    print("p2_v6_hold_release_test: PASS")


if __name__ == "__main__":
    main()
