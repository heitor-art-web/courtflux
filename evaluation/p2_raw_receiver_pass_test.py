from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from evaluation.p2_event_engine_test import ball, frame_state, player
from services.p2 import P2EngineV1
from services.p2.possession_tracker_v1 import PossessionTrackerV1


def missing_ball(frame_id: int) -> BallState:
    return BallState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        status=BallStateStatus.MISSING,
        source="synthetic_test",
    )


class P2RawReceiverPassTest(unittest.TestCase):
    @staticmethod
    def _engine() -> P2EngineV1:
        engine = P2EngineV1(fps=10.0)
        engine.possession_tracker = PossessionTrackerV1(
            lock_frames=1,
            lost_grace_frames=1,
            loose_confirm_frames=1,
            release_confirm_frames=2,
            pass_min_direct_observations=2,
            pass_min_displacement_heights=0.65,
        )
        return engine

    @staticmethod
    def _seed_ownerless_flight(engine: P2EngineV1) -> None:
        owner = player(0, 1, (10, 180, 30, 220))
        engine.process(
            frame_state(
                0,
                player_states=[owner],
                ball_state=ball(0, 20, 200),
            )
        )
        engine.process(frame_state(1, player_states=[], ball_state=missing_ball(1)))
        engine.process(frame_state(2, player_states=[], ball_state=missing_ball(2)))
        self_diag = engine.possession_tracker.diagnostics()
        assert self_diag["flight"]["active"]

    def test_persistent_raw_best_plus_convergence_can_classify_pass(self) -> None:
        engine = self._engine()
        self._seed_ownerless_flight(engine)

        # The receiver is deliberately far enough from the ball to stay below the normal
        # possession acquire threshold. It is still the raw-best player on both frames.
        receiver3 = player(3, 2, (170, 180, 190, 220))
        first = frame_state(
            3,
            player_states=[receiver3],
            ball_state=ball(3, 80, 200),
        )
        engine.process(first)
        first_possession = first.diagnostics["p2"]["possession"]
        self.assertIsNone(first_possession["candidate_track_id"])
        self.assertEqual(first_possession["raw_best_track_id"], 2)
        self.assertNotEqual(first_possession["control_state"], "pass_flight")

        receiver4 = player(4, 2, (170, 180, 190, 220))
        classified = frame_state(
            4,
            player_states=[receiver4],
            ball_state=ball(4, 120, 200),
        )
        engine.process(classified)

        possession = classified.diagnostics["p2"]["possession"]
        live = classified.diagnostics["p2"]["live_control"]
        self.assertIsNone(possession["candidate_track_id"])
        self.assertEqual(possession["raw_best_track_id"], 2)
        self.assertEqual(possession["control_state"], "pass_flight")
        self.assertEqual(possession["live_control_source"], "raw_receiver_trajectory_pass")
        self.assertTrue(live["raw_receiver"]["raw_best_same_track_consecutive"])
        self.assertTrue(live["raw_receiver"]["raw_receiver_eligible"])
        self.assertTrue(live["receiver_trajectory"]["converging"])
        self.assertEqual(
            live["receiver_trajectory"]["receiver_evidence_source"],
            "raw_best_temporal",
        )

    def test_changed_raw_best_identity_does_not_classify_pass(self) -> None:
        engine = self._engine()
        self._seed_ownerless_flight(engine)

        receiver3 = player(3, 2, (170, 180, 190, 220))
        engine.process(
            frame_state(
                3,
                player_states=[receiver3],
                ball_state=ball(3, 80, 200),
            )
        )

        receiver4 = player(4, 3, (170, 180, 190, 220))
        state = frame_state(
            4,
            player_states=[receiver4],
            ball_state=ball(4, 120, 200),
        )
        engine.process(state)

        possession = state.diagnostics["p2"]["possession"]
        live = state.diagnostics["p2"]["live_control"]
        self.assertEqual(possession["raw_best_track_id"], 3)
        self.assertFalse(live["raw_receiver"]["raw_best_same_track_consecutive"])
        self.assertFalse(live["raw_receiver"]["raw_receiver_eligible"])
        self.assertNotEqual(possession["control_state"], "pass_flight")

    def test_shot_first_still_preempts_raw_receiver_path(self) -> None:
        engine = self._engine()
        self._seed_ownerless_flight(engine)

        # Above the rim and inside the frozen <=8 normalized-distance preflight, while
        # outside ShotEventTrackerV1's near_x<=2.5 rim-width trajectory start. A raw-best
        # receiver is present, but SHOT must remain the first classifier.
        receiver3 = player(3, 2, (190, 40, 210, 100))
        first = frame_state(
            3,
            player_states=[receiver3],
            ball_state=ball(3, 160, 70),
        )
        engine.process(first)
        self.assertEqual(
            first.diagnostics["p2"]["possession"]["control_state"],
            "shot_flight",
        )

        receiver4 = player(4, 2, (190, 40, 210, 100))
        second = frame_state(
            4,
            player_states=[receiver4],
            ball_state=ball(4, 170, 70),
        )
        engine.process(second)
        possession = second.diagnostics["p2"]["possession"]
        self.assertEqual(possession["control_state"], "shot_flight")
        self.assertNotEqual(possession["live_control_source"], "raw_receiver_trajectory_pass")


if __name__ == "__main__":
    unittest.main()
