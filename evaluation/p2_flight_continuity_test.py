from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import BallControlState
from core.player_state import PlayerState
from services.p2.possession_tracker_v1 import PossessionTrackerV1


def player(frame_id: int, track_id: int, box: tuple[int, int, int, int]) -> PlayerState:
    x1, y1, x2, y2 = box
    return PlayerState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 25.0,
        track_id=track_id,
        bbox=box,
        center_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        confidence=0.95,
        uncertainty=0.05,
    )


def ball(frame_id: int, x: float, y: float) -> BallState:
    return BallState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 25.0,
        status=BallStateStatus.OBSERVED,
        x_px=x,
        y_px=y,
        confidence=0.95,
        uncertainty=0.05,
        trajectory_score=0.9,
        source="synthetic_test",
    )


class P2FlightContinuityTest(unittest.TestCase):
    def test_open_flight_keeps_observing_after_handler_grace_expires(self) -> None:
        tracker = PossessionTrackerV1(
            lock_frames=1,
            lost_grace_frames=1,
            loose_confirm_frames=1,
            pass_min_direct_observations=3,
            pass_min_displacement_heights=0.65,
        )
        tracker.update(
            frame_id=0,
            players=[player(0, 1, (10, 10, 30, 50))],
            ball_state=ball(0, 20, 30),
        )
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)

        tracker.update(frame_id=1, players=[], ball_state=ball(1, 45, 30))
        tracker.update(frame_id=2, players=[], ball_state=ball(2, 60, 30))
        self.assertIsNone(tracker.current_track_id)
        self.assertEqual(tracker.diagnostics()["flight"]["flight_direct_observations"], 2)
        self.assertFalse(tracker.diagnostics()["flight"]["flight_supported"])

        tracker.update(frame_id=3, players=[], ball_state=ball(3, 75, 30))
        flight = tracker.diagnostics()["flight"]
        self.assertEqual(flight["flight_direct_observations"], 3)
        self.assertGreaterEqual(flight["flight_frames"], 3)
        self.assertTrue(flight["flight_supported"])
        self.assertEqual(tracker.control_state, BallControlState.PASS_FLIGHT)

    def test_open_flight_observes_while_new_handler_lock_is_pending(self) -> None:
        tracker = PossessionTrackerV1(
            lock_frames=2,
            lost_grace_frames=1,
            loose_confirm_frames=1,
            pass_min_direct_observations=4,
            pass_min_displacement_heights=0.65,
        )
        owner = player(0, 1, (10, 10, 30, 50))
        tracker.update(frame_id=0, players=[owner], ball_state=ball(0, 20, 30))
        tracker.update(frame_id=1, players=[owner], ball_state=ball(1, 20, 30))
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)

        tracker.update(frame_id=2, players=[], ball_state=ball(2, 50, 30))
        tracker.update(frame_id=3, players=[], ball_state=ball(3, 65, 30))
        self.assertIsNone(tracker.current_track_id)

        receiver = player(4, 2, (70, 10, 90, 50))
        tracker.update(frame_id=4, players=[receiver], ball_state=ball(4, 80, 30))
        flight = tracker.diagnostics()["flight"]
        self.assertEqual(flight["flight_direct_observations"], 3)
        self.assertIsNone(tracker.current_track_id)

        receiver = player(5, 2, (75, 10, 95, 50))
        tracker.update(frame_id=5, players=[receiver], ball_state=ball(5, 85, 30))
        self.assertEqual(tracker.current_track_id, 2)


if __name__ == "__main__":
    unittest.main()
