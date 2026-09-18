from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import ShotOutcome
from core.court_state import CourtState
from services.p2.shot_event_tracker_v1 import ShotEventTrackerV1


def ball(frame_id: int, x: float, y: float) -> BallState:
    return BallState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        status=BallStateStatus.OBSERVED,
        x_px=x,
        y_px=y,
        confidence=0.95,
        uncertainty=0.05,
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


class DeferredOutcomeResetRegressionTest(unittest.TestCase):
    def test_external_reset_clears_deferred_rim_exit_state(self) -> None:
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
            ball_state=ball(1, 100.0, 90.0),
            court_state=court(1),
        )
        tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=BallState(
                frame_id=2,
                timestamp_sec=0.2,
                status=BallStateStatus.PREDICTED,
                x_px=120.0,
                y_px=100.0,
                confidence=0.72,
                uncertainty=0.28,
                trajectory_score=0.90,
                source="synthetic_test",
            ),
            court_state=court(2),
        )
        self.assertEqual(
            tracker.update(
                frame_id=3,
                timestamp_sec=0.3,
                ball_state=ball(3, 140.0, 104.0),
                court_state=court(3),
            ),
            [],
        )
        self.assertTrue(tracker.diagnostics()["rim_exit_deferred"])

        tracker.reset(reason="test_reset")
        diagnostics = tracker.diagnostics()
        self.assertFalse(diagnostics["rim_exit_deferred"])
        self.assertIsNone(diagnostics["rim_exit_deferred_frame"])
        self.assertIsNone(diagnostics["rim_exit_deferred_deadline_frame"])
        self.assertEqual(diagnostics["last_reset_reason"], "test_reset")


if __name__ == "__main__":
    unittest.main()
