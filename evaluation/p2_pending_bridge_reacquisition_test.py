from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import ShotOutcome
from core.court_state import CourtState
from services.p2.shot_event_tracker_v1 import ShotEventTrackerV1


def ball(
    frame_id: int,
    x: float,
    y: float,
    *,
    status: BallStateStatus = BallStateStatus.OBSERVED,
) -> BallState:
    return BallState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        status=status,
        x_px=x,
        y_px=y,
        confidence=0.95 if status is BallStateStatus.OBSERVED else 0.72,
        uncertainty=0.05 if status is BallStateStatus.OBSERVED else 0.28,
        trajectory_score=0.90,
        source="synthetic_test",
    )


def court(frame_id: int) -> CourtState:
    return CourtState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        mode="image_space_fallback",
        court_visible=True,
        analysis_eligible=True,
        basket_evidence=[
            {
                "kind": "rim",
                "bbox_px": [90.0, 90.0, 110.0, 100.0],
                "center_px": [100.0, 95.0],
                "confidence": 0.95,
                "event_truth": False,
            }
        ],
    )


class PendingBridgeReacquisitionRegressionTest(unittest.TestCase):
    def test_far_post_gap_reacquisition_cannot_seed_pending_bridge_and_stays_unknown(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 100.0, 70.0),
            court_state=court(0),
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 100.0, 88.0),
            court_state=court(1),
        )
        tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(
                2,
                100.0,
                96.0,
                status=BallStateStatus.PREDICTED,
            ),
            court_state=court(2),
        )

        # The long segment from (100, 88) to (170, 150) intersects the rim
        # mouth mathematically, but the direct post-gap point is too far from
        # the rim to be trusted as a bridge endpoint. The revised policy may
        # defer the ambiguous rim exit, but must never promote it to MADE.
        self.assertEqual(
            tracker.update(
                frame_id=3,
                timestamp_sec=0.3,
                ball_state=ball(3, 170.0, 150.0),
                court_state=court(3),
            ),
            [],
        )
        self.assertFalse(tracker.diagnostics()["pending_bridge"])
        self.assertTrue(tracker.diagnostics()["rim_exit_deferred"])

        for frame_id, x, y in ((4, 172.0, 152.0), (5, 174.0, 154.0)):
            self.assertEqual(
                tracker.update(
                    frame_id=frame_id,
                    timestamp_sec=frame_id / 10.0,
                    ball_state=ball(frame_id, x, y),
                    court_state=court(frame_id),
                ),
                [],
            )

        events = tracker.update(
            frame_id=6,
            timestamp_sec=0.6,
            ball_state=ball(6, 176.0, 156.0),
            court_state=court(6),
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertEqual(
            event.attributes["resolution_reason"],
            "deferred_rim_exit_expired_without_confirmation",
        )
        self.assertFalse(event.attributes["pending_bridge_used"])
        self.assertIsNone(event.attributes["post_gap_fall_ratio"])
        self.assertIn(
            "pending_bridge_rejected_reacquisition_not_near_rim",
            event.evidence,
        )


if __name__ == "__main__":
    unittest.main()
