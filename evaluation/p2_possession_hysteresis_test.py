from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import BallControlState, BasketballEventType
from core.player_state import PlayerState
from services.p2.possession_tracker_v1 import PossessionTrackerV1


def player(frame_id: int, track_id: int, box: tuple[int, int, int, int]) -> PlayerState:
    x1, y1, x2, y2 = box
    return PlayerState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        track_id=track_id,
        bbox=box,
        center_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        confidence=0.95,
        uncertainty=0.05,
    )


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


class P2PossessionHysteresisTest(unittest.TestCase):
    def test_same_handler_below_threshold_stays_possessed_inside_existing_grace(self) -> None:
        tracker = PossessionTrackerV1(
            lock_frames=1,
            switch_frames=1,
            lost_grace_frames=2,
            loose_confirm_frames=2,
        )
        p0 = player(0, 1, (10, 10, 30, 50))
        acquired = tracker.update(frame_id=0, players=[p0], ball_state=ball(0, 20, 30))
        self.assertIn(
            BasketballEventType.POSSESSION_ACQUIRED,
            {event.event_type for event in acquired},
        )
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)

        # The same player remains the best-ranked handler, but the ball is far enough
        # away for the score to fall below switch_threshold. This should consume the
        # existing grace window without inventing PASS_FLIGHT or LOOSE.
        events = tracker.update(
            frame_id=1,
            players=[player(1, 1, (10, 10, 30, 50))],
            ball_state=ball(1, 45, 30),
        )
        self.assertEqual(events, [])
        self.assertEqual(tracker.current_track_id, 1)
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)
        self.assertEqual(tracker.diagnostics()["lost_frames"], 1)

        events = tracker.update(
            frame_id=2,
            players=[player(2, 1, (10, 10, 30, 50))],
            ball_state=ball(2, 45, 30),
        )
        self.assertEqual(events, [])
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)
        self.assertEqual(tracker.diagnostics()["lost_frames"], 2)

    def test_existing_release_logic_resumes_after_grace_is_exhausted(self) -> None:
        tracker = PossessionTrackerV1(
            lock_frames=1,
            switch_frames=1,
            lost_grace_frames=1,
            loose_confirm_frames=2,
            pass_min_displacement_heights=2.0,
        )
        tracker.update(
            frame_id=0,
            players=[player(0, 1, (10, 10, 30, 50))],
            ball_state=ball(0, 20, 30),
        )

        first = tracker.update(
            frame_id=1,
            players=[player(1, 1, (10, 10, 30, 50))],
            ball_state=ball(1, 45, 30),
        )
        self.assertEqual(first, [])
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)

        second = tracker.update(
            frame_id=2,
            players=[player(2, 1, (10, 10, 30, 50))],
            ball_state=ball(2, 45, 30),
        )
        self.assertIn(
            BasketballEventType.LOOSE_BALL_STARTED,
            {event.event_type for event in second},
        )
        self.assertEqual(tracker.control_state, BallControlState.LOOSE)


if __name__ == "__main__":
    unittest.main()
