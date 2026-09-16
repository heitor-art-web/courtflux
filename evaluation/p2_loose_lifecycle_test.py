from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.basketball_event import BasketballEventType
from evaluation.p2_event_engine_test import ball, frame_state, player
from services.p2.engine_v1 import P2EngineV1
from services.p2.metrics import P2MetricsCollector
from services.p2.possession_tracker_v1 import PossessionTrackerV1


class P2LooseLifecycleTest(unittest.TestCase):
    @staticmethod
    def _engine(*, reset_after_ineligible_frames: int = 3) -> P2EngineV1:
        engine = P2EngineV1(
            fps=10.0,
            reset_after_ineligible_frames=reset_after_ineligible_frames,
        )
        engine.possession_tracker = PossessionTrackerV1(
            lock_frames=1,
            switch_frames=1,
            lost_grace_frames=10,
            loose_confirm_frames=2,
            pass_min_direct_observations=99,
        )
        return engine

    @staticmethod
    def _enter_loose(engine: P2EngineV1) -> None:
        teams = {1: ("blue", 0.95), 2: ("red", 0.95)}
        engine.process(
            frame_state(
                0,
                player_states=[player(0, 1, (10, 10, 30, 50))],
                ball_state=ball(0, 20, 30),
                teams=teams,
            )
        )
        engine.process(
            frame_state(
                1,
                player_states=[player(1, 1, (10, 10, 30, 50))],
                ball_state=ball(1, 45, 30),
                teams=teams,
            )
        )
        loose = frame_state(
            2,
            player_states=[player(2, 1, (10, 10, 30, 50))],
            ball_state=ball(2, 45, 30),
            teams=teams,
        )
        engine.process(loose)
        self_types = {event.event_type for event in loose.p2_events}
        if BasketballEventType.LOOSE_BALL_STARTED not in self_types:
            raise AssertionError("fixture failed to enter LOOSE state")

    def test_new_handler_switch_closes_loose_interval(self) -> None:
        engine = self._engine()
        self._enter_loose(engine)
        teams = {1: ("blue", 0.95), 2: ("red", 0.95)}

        switched = frame_state(
            3,
            player_states=[
                player(3, 1, (10, 10, 30, 50)),
                player(3, 2, (40, 10, 50, 50)),
            ],
            ball_state=ball(3, 45, 30),
            teams=teams,
        )
        engine.process(switched)

        event_types = {event.event_type for event in switched.p2_events}
        self.assertIn(BasketballEventType.POSSESSION_CHANGE, event_types)
        self.assertIn(BasketballEventType.LOOSE_BALL_RECOVERED, event_types)
        recovery = next(
            event
            for event in switched.p2_events
            if event.event_type is BasketballEventType.LOOSE_BALL_RECOVERED
        )
        self.assertEqual(recovery.actor_track_id, 2)
        self.assertEqual(recovery.secondary_track_id, 1)
        self.assertEqual(recovery.attributes["terminal_reason"], "new_handler_switch")
        self.assertFalse(recovery.attributes["same_handler_recovery"])
        self.assertTrue(recovery.attributes["recovered"])

    def test_ineligible_reset_emits_terminal_loose_event(self) -> None:
        engine = self._engine(reset_after_ineligible_frames=2)
        self._enter_loose(engine)

        engine.process(frame_state(3, eligible=False))
        terminal = frame_state(4, eligible=False)
        engine.process(terminal)

        ended = next(
            event
            for event in terminal.p2_events
            if event.event_type is BasketballEventType.LOOSE_BALL_ENDED
        )
        self.assertEqual(
            ended.attributes["terminal_reason"],
            "reset:gameplay_absent_gap",
        )
        self.assertFalse(ended.attributes["recovered"])
        self.assertEqual(
            terminal.diagnostics["p2"]["events"][0]["event_type"],
            BasketballEventType.LOOSE_BALL_ENDED.value,
        )

    def test_shot_start_emits_terminal_loose_event(self) -> None:
        engine = self._engine()
        self._enter_loose(engine)
        teams = {1: ("blue", 0.95)}

        shot_start = frame_state(
            3,
            player_states=[player(3, 1, (10, 10, 30, 50))],
            ball_state=ball(3, 100, 70),
            teams=teams,
        )
        engine.process(shot_start)

        ended = next(
            event
            for event in shot_start.p2_events
            if event.event_type is BasketballEventType.LOOSE_BALL_ENDED
        )
        self.assertEqual(ended.attributes["terminal_reason"], "shot_flight_started")
        self.assertFalse(ended.attributes["recovered"])

    def test_probe_reports_loose_closure_and_terminal_reasons(self) -> None:
        collector = P2MetricsCollector()
        frame = SimpleNamespace(
            frame_id=10,
            timestamp=1.0,
            fps=10.0,
            valid_gameplay=True,
            court_state=None,
            diagnostics={
                "p2": {
                    "engine_eligible": True,
                    "p1_state_is_not_mutated": True,
                    "possession": {
                        "control_state": "unknown",
                        "current_track_id": None,
                        "current_team_id": None,
                        "current_confidence": 0.0,
                        "lost_frames": 0,
                        "flight": {},
                    },
                    "shot": {},
                    "temporal_reset": {"count": 0},
                    "events": [
                        {
                            "event_type": "loose_ball_started",
                            "decision": "candidate",
                            "confidence": 0.6,
                            "actor_track_id": 1,
                            "team_id": None,
                            "attributes": {},
                        },
                        {
                            "event_type": "loose_ball_ended",
                            "decision": "confirmed",
                            "confidence": 1.0,
                            "actor_track_id": 1,
                            "team_id": None,
                            "attributes": {
                                "terminal_reason": "reset:frame_discontinuity",
                                "recovered": False,
                            },
                        },
                    ],
                }
            },
        )
        collector.observe(frame)
        transitions = collector.summary()["transitions"]
        self.assertEqual(transitions["loose_ball_started"], 1)
        self.assertEqual(transitions["loose_ball_ended"], 1)
        self.assertEqual(transitions["loose_ball_closed"], 1)
        self.assertEqual(transitions["loose_ball_open_intervals"], 0)
        self.assertEqual(transitions["loose_ball_closure_rate"], 100.0)
        self.assertEqual(
            transitions["loose_ball_terminal_reasons"],
            {"reset:frame_discontinuity": 1},
        )


if __name__ == "__main__":
    unittest.main()
