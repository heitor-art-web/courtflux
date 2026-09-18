from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from evaluation.p2_event_engine_test import ball, frame_state, player
from services.p2 import P2EngineV1
from services.p2.engine_v1 import P2EngineV1 as DirectP2EngineV1
from services.p2.possession_tracker_v1 import PossessionTrackerV1


def missing_ball(frame_id: int) -> BallState:
    return BallState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        status=BallStateStatus.MISSING,
        source="synthetic_test",
    )


class P2LiveControlResolverTest(unittest.TestCase):
    def test_direct_engine_import_uses_canonical_live_resolver(self) -> None:
        self.assertIs(P2EngineV1, DirectP2EngineV1)
        engine = P2EngineV1(fps=10.0)
        self.assertEqual(engine.__class__.__name__, "LiveControlP2EngineV1")

    @staticmethod
    def _owner_expiry_engine() -> P2EngineV1:
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
    def _pass_engine() -> P2EngineV1:
        engine = P2EngineV1(fps=10.0)
        engine.possession_tracker = PossessionTrackerV1(
            lock_frames=3,
            lost_grace_frames=1,
            loose_confirm_frames=1,
            release_confirm_frames=2,
            pass_min_direct_observations=2,
            pass_min_displacement_heights=0.65,
        )
        for frame_id in range(3):
            owner = player(frame_id, 1, (10, 180, 30, 220))
            engine.process(
                frame_state(
                    frame_id,
                    player_states=[owner],
                    ball_state=ball(frame_id, 20, 200),
                )
            )
        engine.process(frame_state(3, player_states=[], ball_state=missing_ball(3)))
        engine.process(frame_state(4, player_states=[], ball_state=missing_ball(4)))
        return engine

    def test_recent_owner_continuity_replaces_unknown_with_possessed(self) -> None:
        engine = self._owner_expiry_engine()
        owner = player(0, 1, (10, 10, 30, 50))
        first = frame_state(0, player_states=[owner], ball_state=ball(0, 20, 30))
        engine.process(first)
        self.assertEqual(first.diagnostics["p2"]["possession"]["control_state"], "possessed")

        engine.process(frame_state(1, player_states=[], ball_state=missing_ball(1)))
        expired = frame_state(2, player_states=[], ball_state=missing_ball(2))
        engine.process(expired)

        possession = expired.diagnostics["p2"]["possession"]
        self.assertIsNone(engine.possession_tracker.current_track_id)
        self.assertEqual(possession["tracker_control_state"], "unknown")
        self.assertEqual(possession["control_state"], "possessed")
        self.assertEqual(
            possession["live_control_source"],
            "airborne_unclassified_owner_continuity",
        )
        self.assertEqual(possession["internal_airborne_state"], "airborne_unclassified")
        self.assertTrue(possession["flight"]["active"])
        self.assertTrue(possession["flight_seeded_from_recent_owner"])

    def test_frozen_preflight_promotes_and_latches_shot_flight(self) -> None:
        engine = self._owner_expiry_engine()
        owner = player(0, 1, (10, 10, 30, 50))
        engine.process(frame_state(0, player_states=[owner], ball_state=ball(0, 20, 30)))
        engine.process(frame_state(1, player_states=[], ball_state=missing_ball(1)))
        engine.process(frame_state(2, player_states=[], ball_state=missing_ball(2)))

        # Above the rim and inside the frozen <=8 distance, but outside the shot tracker's
        # near_x<=2.5-width trajectory start, so this isolates the resolver preflight path.
        preflight = frame_state(3, player_states=[], ball_state=ball(3, 170, 70))
        engine.process(preflight)
        possession = preflight.diagnostics["p2"]["possession"]
        live = preflight.diagnostics["p2"]["live_control"]
        self.assertEqual(possession["tracker_control_state"], "unknown")
        self.assertEqual(possession["control_state"], "shot_flight")
        self.assertEqual(possession["live_control_source"], "frozen_shot_preflight")
        self.assertEqual(possession["internal_airborne_state"], "shot_flight")
        self.assertTrue(live["shot_preflight"]["signal"])
        self.assertFalse(live["shot_active"])

        latched = frame_state(4, player_states=[], ball_state=ball(4, 250, 250))
        engine.process(latched)
        possession = latched.diagnostics["p2"]["possession"]
        self.assertEqual(possession["control_state"], "shot_flight")
        self.assertEqual(possession["live_control_source"], "latched_shot_flight")

    def test_shot_preflight_has_precedence_over_receiver_candidate(self) -> None:
        engine = P2EngineV1(fps=10.0)
        engine.possession_tracker = PossessionTrackerV1(
            lock_frames=3,
            lost_grace_frames=1,
            loose_confirm_frames=1,
            release_confirm_frames=2,
            pass_min_direct_observations=2,
            pass_min_displacement_heights=0.65,
        )
        for frame_id in range(3):
            owner = player(frame_id, 1, (10, 10, 30, 50))
            engine.process(
                frame_state(
                    frame_id,
                    player_states=[owner],
                    ball_state=ball(frame_id, 20, 30),
                )
            )
        engine.process(frame_state(3, player_states=[], ball_state=missing_ball(3)))
        engine.process(frame_state(4, player_states=[], ball_state=missing_ball(4)))

        # x=160 is outside ShotEventTrackerV1 near_x, but remains within frozen distance 8.
        receiver = player(5, 2, (145, 45, 185, 95))
        state = frame_state(5, player_states=[receiver], ball_state=ball(5, 160, 70))
        engine.process(state)
        possession = state.diagnostics["p2"]["possession"]
        live = state.diagnostics["p2"]["live_control"]
        self.assertTrue(live["receiver_evidence"])
        self.assertTrue(live["shot_preflight"]["signal"])
        self.assertEqual(possession["control_state"], "shot_flight")
        self.assertEqual(possession["live_control_source"], "frozen_shot_preflight")

    def test_distinct_receiver_pass_requires_converging_trajectory_and_latches(self) -> None:
        engine = self._pass_engine()

        # Both samples are inside the receiver box (so possession candidate evidence is
        # strong), while their centers show the ball moving 20px -> 10px from the receiver.
        receiver5 = player(5, 2, (70, 180, 130, 220))
        first = frame_state(5, player_states=[receiver5], ball_state=ball(5, 80, 200))
        engine.process(first)
        self.assertNotEqual(
            first.diagnostics["p2"]["possession"]["control_state"],
            "pass_flight",
        )

        receiver6 = player(6, 2, (70, 180, 130, 220))
        classified = frame_state(
            6,
            player_states=[receiver6],
            ball_state=ball(6, 90, 200),
        )
        engine.process(classified)
        possession = classified.diagnostics["p2"]["possession"]
        trajectory = classified.diagnostics["p2"]["live_control"]["receiver_trajectory"]
        self.assertTrue(trajectory["converging"])
        self.assertEqual(possession["control_state"], "pass_flight")
        self.assertEqual(possession["live_control_source"], "receiver_trajectory_pass")
        self.assertEqual(possession["internal_airborne_state"], "pass_flight")

        latched = frame_state(7, player_states=[], ball_state=ball(7, 95, 200))
        engine.process(latched)
        possession = latched.diagnostics["p2"]["possession"]
        self.assertEqual(possession["tracker_control_state"], "unknown")
        self.assertEqual(possession["control_state"], "pass_flight")
        self.assertEqual(possession["live_control_source"], "latched_pass_flight")

    def test_receiver_without_converging_trajectory_stays_airborne_unclassified(self) -> None:
        engine = self._pass_engine()
        receiver5 = player(5, 2, (70, 180, 130, 220))
        engine.process(
            frame_state(5, player_states=[receiver5], ball_state=ball(5, 90, 200))
        )
        receiver6 = player(6, 2, (70, 180, 130, 220))
        state = frame_state(6, player_states=[receiver6], ball_state=ball(6, 80, 200))
        engine.process(state)

        possession = state.diagnostics["p2"]["possession"]
        live = state.diagnostics["p2"]["live_control"]
        self.assertTrue(live["receiver_evidence"])
        self.assertFalse(live["receiver_trajectory"]["converging"])
        self.assertEqual(possession["internal_airborne_state"], "airborne_unclassified")
        self.assertEqual(possession["control_state"], "possessed")
        self.assertEqual(
            possession["live_control_source"],
            "airborne_unclassified_owner_continuity",
        )

    def test_gameplay_grace_holds_state_then_resets_without_processing_events(self) -> None:
        engine = P2EngineV1(fps=10.0)
        for frame_id in range(3):
            owner = player(frame_id, 1, (10, 10, 30, 50))
            engine.process(
                frame_state(
                    frame_id,
                    player_states=[owner],
                    ball_state=ball(frame_id, 20, 30),
                )
            )
        self.assertEqual(engine.possession_tracker.current_track_id, 1)
        self.assertEqual(engine.gameplay_continuity_grace_frames, 9)

        last_grace = None
        for frame_id in range(3, 12):
            state = frame_state(frame_id, eligible=False)
            engine.process(state)
            last_grace = state
            self.assertEqual(state.p2_events, [])
            self.assertEqual(
                state.diagnostics["p2"]["possession"]["control_state"],
                "possessed",
            )
            self.assertTrue(state.diagnostics["p2"]["engine_eligible"])
            self.assertEqual(
                state.diagnostics["p2"]["skipped_reason"],
                "gameplay_continuity_grace",
            )
        self.assertIsNotNone(last_grace)
        self.assertEqual(engine.possession_tracker.current_track_id, 1)

        expired = frame_state(12, eligible=False)
        engine.process(expired)
        self.assertIsNone(engine.possession_tracker.current_track_id)
        self.assertFalse(expired.diagnostics["p2"]["engine_eligible"])
        self.assertEqual(
            expired.diagnostics["p2"]["possession"]["control_state"],
            "unknown",
        )
        self.assertEqual(
            expired.diagnostics["p2"]["temporal_reset"]["last_reason"],
            "gameplay_absent_gap",
        )

    def test_explicit_reset_after_ineligible_frames_keeps_legacy_contract(self) -> None:
        engine = P2EngineV1(fps=10.0, reset_after_ineligible_frames=2)
        self.assertEqual(engine.gameplay_continuity_grace_frames, 1)
        for frame_id in range(3):
            owner = player(frame_id, 1, (10, 10, 30, 50))
            engine.process(
                frame_state(
                    frame_id,
                    player_states=[owner],
                    ball_state=ball(frame_id, 20, 30),
                )
            )

        first_gap = frame_state(3, eligible=False)
        engine.process(first_gap)
        self.assertTrue(first_gap.diagnostics["p2"]["engine_eligible"])

        second_gap = frame_state(4, eligible=False)
        engine.process(second_gap)
        self.assertFalse(second_gap.diagnostics["p2"]["engine_eligible"])
        self.assertIsNone(engine.possession_tracker.current_track_id)
        self.assertEqual(
            second_gap.diagnostics["p2"]["temporal_reset"]["last_reason"],
            "gameplay_absent_gap",
        )


if __name__ == "__main__":
    unittest.main()
