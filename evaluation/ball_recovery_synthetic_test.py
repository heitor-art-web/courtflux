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
        high_confidence_gate_threshold=0.85,
        high_confidence_gate_multiplier=1.0,
        reacquire_enabled=True,
        reacquire_min_confidence=0.45,
        reacquire_confidence_margin=0.15,
        reacquire_confirm_frames=2,
        reacquire_radius_norm=0.04,
        reacquire_min_size_similarity=0.25,
        reacquire_min_unsupported_frames=2,
        reacquire_promotion_frames=3,
    )


def test_wrong_size_challenger_is_rejected() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    decisions = [
        subject.select(
            [candidate(500 + frame_id, 500, 0.95, size=80)],
            frame_id,
            FRAME_SHAPE,
        )
        for frame_id in (1, 2, 3, 4)
    ]

    assert not any(decision.reacquired for decision in decisions)
    assert all(decision.source != "detector_reacquired" for decision in decisions)


def test_confirmed_challenger_is_provisional_before_promotion() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    first = subject.select([candidate(500, 500, 0.95)], 1, FRAME_SHAPE)
    second = subject.select([candidate(505, 500, 0.95)], 2, FRAME_SHAPE)

    assert first.source == "predicted_hold"
    assert first.challenger_streak == 1
    assert first.recovery_provisional is False
    assert second.source == "predicted_hold"
    assert second.reacquired is False
    assert second.challenger_streak == 2
    assert second.recovery_provisional is True


def test_provisional_recovery_does_not_mutate_active_state() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)
    original_center = subject.last_center.copy()
    original_velocity = subject.velocity.copy()

    subject.select([candidate(500, 500, 0.95)], 1, FRAME_SHAPE)
    provisional = subject.select([candidate(505, 500, 0.95)], 2, FRAME_SHAPE)

    assert provisional.recovery_provisional is True
    assert subject.last_frame_id == 0
    assert subject.last_center is not None
    assert (subject.last_center == original_center).all()
    assert (subject.velocity == original_velocity).all()


def test_active_hold_blocks_strong_recovery_promotion() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    first = subject.select([candidate(500, 500, 0.95)], 1, FRAME_SHAPE)
    second = subject.select([candidate(505, 500, 0.95)], 2, FRAME_SHAPE)
    third = subject.select([candidate(510, 500, 0.95)], 3, FRAME_SHAPE)

    assert first.source == "predicted_hold"
    assert second.recovery_provisional is True
    assert third.source == "predicted_hold"
    assert third.reacquired is False
    assert third.challenger_streak == 3
    assert third.recovery_provisional is True
    assert subject.last_frame_id == 0


def test_strong_recovery_promotes_after_active_hold_expires() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    decisions = [
        subject.select([candidate(500 + frame_id * 5, 500, 0.95)], frame_id, FRAME_SHAPE)
        for frame_id in range(1, 7)
    ]

    assert all(not decision.reacquired for decision in decisions[:5])
    assert decisions[4].source == "predicted_hold"
    assert decisions[5].source == "detector_reacquired"
    assert decisions[5].reacquired is True


def test_off_frame_active_prediction_allows_strong_promotion() -> None:
    subject = tracker()
    subject.select([candidate(980, 500, 0.60)], 0, FRAME_SHAPE)
    subject.velocity[:] = [20.0, 0.0]

    first = subject.select([candidate(500, 500, 0.95)], 1, FRAME_SHAPE)
    second = subject.select([candidate(505, 500, 0.95)], 2, FRAME_SHAPE)
    third = subject.select([candidate(510, 500, 0.95)], 3, FRAME_SHAPE)

    assert first.source == "predicted_hold"
    assert second.recovery_provisional is True
    assert third.source == "detector_reacquired"
    assert third.reacquired is True


def test_weak_recovery_does_not_promote_when_active_invalid() -> None:
    subject = tracker()
    subject.select([candidate(980, 500, 0.60)], 0, FRAME_SHAPE)
    subject.velocity[:] = [20.0, 0.0]

    decisions = [
        subject.select([candidate(500 + frame_id * 5, 500, 0.80)], frame_id, FRAME_SHAPE)
        for frame_id in (1, 2, 3, 4)
    ]

    assert decisions[1].recovery_provisional is True
    assert not any(decision.reacquired for decision in decisions)


def test_promoted_recovery_uses_challenger_velocity() -> None:
    subject = tracker()
    subject.select([candidate(980, 500, 0.60)], 0, FRAME_SHAPE)
    subject.velocity[:] = [20.0, 0.0]

    subject.select([candidate(500, 500, 0.95)], 1, FRAME_SHAPE)
    subject.select([candidate(510, 500, 0.95)], 2, FRAME_SHAPE)
    promoted = subject.select([candidate(520, 500, 0.95)], 3, FRAME_SHAPE)
    hold = subject.select([], 4, FRAME_SHAPE)

    assert promoted.source == "detector_reacquired"
    assert promoted.challenger_velocity is not None
    assert promoted.challenger_velocity[0] > 0.0
    assert subject.velocity[0] > 0.0
    assert hold.source == "predicted_hold"
    assert hold.ball is not None
    assert float(hold.ball["center"][0]) > 520.0


def test_plausible_association_has_absolute_priority() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    decisions = []
    for frame_id in (1, 2, 3):
        decisions.append(
            subject.select(
                [
                    candidate(100 + frame_id * 2, 100, 0.55),
                    candidate(500 + frame_id * 4, 500, 0.95),
                ],
                frame_id,
                FRAME_SHAPE,
            )
        )

    assert all(decision.source == "detector_associated" for decision in decisions)
    assert all(decision.selected_index == 0 for decision in decisions)
    assert not any(decision.reacquired for decision in decisions)
    assert all(decision.challenger_streak == 0 for decision in decisions)
    assert subject.track_state == "TRACKED"


def test_plausible_association_cancels_provisional_recovery() -> None:
    subject = tracker()
    subject.select([candidate(100, 100, 0.60)], 0, FRAME_SHAPE)

    subject.select([candidate(500, 500, 0.95)], 1, FRAME_SHAPE)
    provisional = subject.select([candidate(505, 500, 0.95)], 2, FRAME_SHAPE)
    supported = subject.select(
        [
            candidate(102, 100, 0.55),
            candidate(510, 500, 0.95),
        ],
        3,
        FRAME_SHAPE,
    )
    after_reset = subject.select([candidate(515, 500, 0.95)], 4, FRAME_SHAPE)

    assert provisional.recovery_provisional is True
    assert supported.source == "detector_associated"
    assert supported.selected_index == 0
    assert supported.reacquired is False
    assert supported.challenger_streak == 0
    assert subject._challenger_provisional is False
    assert after_reset.source == "predicted_hold"
    assert after_reset.reacquired is False
    assert after_reset.challenger_streak == 1


def test_pending_challenger_survives_one_detector_gap() -> None:
    subject = tracker()
    subject.select([candidate(980, 500, 0.60)], 0, FRAME_SHAPE)
    subject.velocity[:] = [20.0, 0.0]

    first = subject.select([candidate(500, 500, 0.95)], 1, FRAME_SHAPE)
    gap = subject.select([], 2, FRAME_SHAPE)
    second = subject.select([candidate(508, 500, 0.95)], 3, FRAME_SHAPE)
    promoted = subject.select([candidate(512, 500, 0.95)], 4, FRAME_SHAPE)

    assert first.challenger_streak == 1
    assert gap.challenger_streak == 1
    assert second.source == "predicted_hold"
    assert second.challenger_streak == 2
    assert second.recovery_provisional is True
    assert promoted.source == "detector_reacquired"
    assert promoted.reacquired is True


def main() -> None:
    tests = [
        test_wrong_size_challenger_is_rejected,
        test_confirmed_challenger_is_provisional_before_promotion,
        test_provisional_recovery_does_not_mutate_active_state,
        test_active_hold_blocks_strong_recovery_promotion,
        test_strong_recovery_promotes_after_active_hold_expires,
        test_off_frame_active_prediction_allows_strong_promotion,
        test_weak_recovery_does_not_promote_when_active_invalid,
        test_promoted_recovery_uses_challenger_velocity,
        test_plausible_association_has_absolute_priority,
        test_plausible_association_cancels_provisional_recovery,
        test_pending_challenger_survives_one_detector_gap,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"[PASS] {len(tests)} recovery state-machine tests")


if __name__ == "__main__":
    main()
