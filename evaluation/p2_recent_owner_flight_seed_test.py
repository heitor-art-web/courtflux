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


def missing_ball(frame_id: int) -> BallState:
    return BallState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 25.0,
        status=BallStateStatus.MISSING,
        source="synthetic_test",
    )


class P2RecentOwnerFlightSeedTest(unittest.TestCase):
    def tracker(self) -> PossessionTrackerV1:
        return PossessionTrackerV1(
            lock_frames=1,
            lost_grace_frames=1,
            loose_confirm_frames=1,
            release_confirm_frames=2,
            pass_min_direct_observations=2,
            pass_min_displacement_heights=0.65,
        )

    def test_owner_expiry_seeds_unsupported_flight_even_when_ball_is_missing(self) -> None:
        tracker = self.tracker()
        owner = player(0, 1, (10, 10, 30, 50))
        tracker.update(frame_id=0, players=[owner], ball_state=ball(0, 20, 30))
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)

        tracker.update(frame_id=1, players=[], ball_state=missing_ball(1))
        self.assertEqual(tracker.current_track_id, 1)
        self.assertFalse(tracker.diagnostics()["flight"]["active"])

        tracker.update(frame_id=2, players=[], ball_state=missing_ball(2))
        diagnostics = tracker.diagnostics()
        self.assertIsNone(tracker.current_track_id)
        self.assertTrue(diagnostics["flight"]["active"])
        self.assertEqual(diagnostics["flight"]["origin_track_id"], 1)
        self.assertEqual(diagnostics["flight"]["flight_direct_observations"], 0)
        self.assertFalse(diagnostics["flight"]["flight_supported"])
        self.assertTrue(diagnostics["flight_seeded_from_recent_owner"])
        self.assertEqual(diagnostics["recent_owner_flight_seed_count"], 1)
        self.assertEqual(tracker.control_state, BallControlState.UNKNOWN)

    def test_seeded_flight_accumulates_existing_evidence_without_becoming_pass(self) -> None:
        tracker = self.tracker()
        owner = player(0, 1, (10, 10, 30, 50))
        tracker.update(frame_id=0, players=[owner], ball_state=ball(0, 20, 30))
        tracker.update(frame_id=1, players=[], ball_state=missing_ball(1))
        tracker.update(frame_id=2, players=[], ball_state=missing_ball(2))

        tracker.update(frame_id=3, players=[], ball_state=ball(3, 50, 30))
        first = tracker.diagnostics()
        self.assertEqual(first["flight"]["flight_direct_observations"], 1)
        self.assertFalse(first["flight"]["flight_supported"])

        tracker.update(frame_id=4, players=[], ball_state=ball(4, 80, 30))
        second = tracker.diagnostics()
        self.assertEqual(second["flight"]["flight_direct_observations"], 2)
        self.assertTrue(second["flight"]["flight_supported"])
        self.assertFalse(second["pass_flight_receiver_evidence"])
        self.assertEqual(second["recent_owner_flight_seed_count"], 1)
        self.assertEqual(tracker.control_state, BallControlState.UNKNOWN)

    def test_existing_live_owner_flight_is_not_seeded_twice(self) -> None:
        tracker = self.tracker()
        owner = player(0, 1, (10, 10, 30, 50))
        tracker.update(frame_id=0, players=[owner], ball_state=ball(0, 20, 30))

        tracker.update(frame_id=1, players=[], ball_state=ball(1, 50, 30))
        self.assertTrue(tracker.diagnostics()["flight"]["active"])
        tracker.update(frame_id=2, players=[], ball_state=ball(2, 80, 30))

        diagnostics = tracker.diagnostics()
        self.assertIsNone(tracker.current_track_id)
        self.assertEqual(diagnostics["recent_owner_flight_seed_count"], 0)
        self.assertFalse(diagnostics["flight_seeded_from_recent_owner"])
        self.assertTrue(diagnostics["flight"]["flight_supported"])
        self.assertEqual(tracker.control_state, BallControlState.UNKNOWN)

    def test_same_owner_reacquisition_clears_seeded_flight(self) -> None:
        tracker = self.tracker()
        owner0 = player(0, 1, (10, 10, 30, 50))
        tracker.update(frame_id=0, players=[owner0], ball_state=ball(0, 20, 30))
        tracker.update(frame_id=1, players=[], ball_state=missing_ball(1))
        tracker.update(frame_id=2, players=[], ball_state=missing_ball(2))
        self.assertTrue(tracker.diagnostics()["flight_seeded_from_recent_owner"])

        owner3 = player(3, 1, (10, 10, 30, 50))
        events = tracker.update(frame_id=3, players=[owner3], ball_state=ball(3, 20, 30))
        self.assertEqual(tracker.current_track_id, 1)
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)
        self.assertFalse(tracker.diagnostics()["flight"]["active"])
        self.assertFalse(
            any(str(event.event_type.value) == "pass_candidate" for event in events)
        )

    def test_no_recent_owner_means_no_ownerless_seed(self) -> None:
        tracker = self.tracker()
        tracker.update(frame_id=0, players=[], ball_state=missing_ball(0))
        tracker.update(frame_id=1, players=[], ball_state=ball(1, 70, 30))
        diagnostics = tracker.diagnostics()
        self.assertFalse(diagnostics["flight"]["active"])
        self.assertEqual(diagnostics["recent_owner_flight_seed_count"], 0)


if __name__ == "__main__":
    unittest.main()
