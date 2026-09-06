from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_tracker import TemporalBallTracker


FRAME_SHAPE = (1000, 1000, 3)


def candidate(
    cx: float,
    cy: float,
    confidence: float,
    size: float = 20.0,
) -> dict[str, object]:
    half = size / 2.0
    return {
        "box": [cx - half, cy - half, cx + half, cy + half],
        "center": (cx, cy),
        "confidence": confidence,
        "interpolated": False,
    }


def tracker() -> TemporalBallTracker:
    return TemporalBallTracker(
        max_hold_frames=5,
        reset_after_frames=12,
        base_gate_norm=0.05,
        gate_growth_per_frame=0.0,
        max_gate_norm=0.05,
        velocity_alpha=0.40,
        score_confidence_weight=0.58,
        score_proximity_weight=0.32,
        score_size_weight=0.10,
        high_confidence_gate_threshold=0.99,
        high_confidence_gate_multiplier=1.0,
        reacquire_enabled=True,
        reacquire_min_confidence=0.45,
        reacquire_confidence_margin=0.15,
        reacquire_confirm_frames=2,
        reacquire_radius_norm=0.04,
        reacquire_min_size_similarity=0.25,
    )


def test_wrong_size_challenger_is_rejected() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    decisions = [
        subject.select([candidate(500 + frame_id, 500, 0.95, size=80)], frame_id, FRAME_SHAPE)
        for frame_id in (1, 2, 3)
    ]

    assert not any(decision.reacquired for decision in decisions)
    assert all(decision.source != "detector_reacquired" for decision in decisions)


def test_degraded_track_can_recover_lower_confidence_candidate() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    first = subject.select([candidate(500, 500, 0.35)], 1, FRAME_SHAPE)
    second = subject.select([candidate(505, 500, 0.35)], 2, FRAME_SHAPE)

    assert first.source == "predicted_hold"
    assert first.track_state == "REACQUIRING"
    assert first.challenger_streak == 1
    assert second.source == "detector_reacquired"
    assert second.reacquired is True


def test_supported_track_requires_confidence_margin() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    decision = subject.select(
        [
            candidate(102, 100, 0.70),
            candidate(500, 500, 0.75),
        ],
        1,
        FRAME_SHAPE,
    )

    assert decision.source == "detector_associated"
    assert decision.selected_index == 0
    assert decision.reacquired is False
    assert decision.challenger_streak == 0


def test_pending_challenger_survives_one_detector_gap() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    first = subject.select([candidate(500, 500, 0.35)], 1, FRAME_SHAPE)
    gap = subject.select([], 2, FRAME_SHAPE)
    recovered = subject.select([candidate(508, 500, 0.35)], 3, FRAME_SHAPE)

    assert first.challenger_streak == 1
    assert gap.challenger_streak == 1
    assert recovered.source == "detector_reacquired"
    assert recovered.reacquired is True


def main() -> None:
    tests = [
        test_wrong_size_challenger_is_rejected,
        test_degraded_track_can_recover_lower_confidence_candidate,
        test_supported_track_requires_confidence_margin,
        test_pending_challenger_survives_one_detector_gap,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] {len(tests)} recovery state-machine tests")


if __name__ == "__main__":
    main()
