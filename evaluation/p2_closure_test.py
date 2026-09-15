from __future__ import annotations

import ast
import unittest
from pathlib import Path

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import BasketballEventType, BallControlState
from core.player_state import PlayerState
from services.p2.possession_tracker_v1 import PossessionTrackerV1


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def player(frame_id: int, track_id: int) -> PlayerState:
    return PlayerState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        track_id=track_id,
        bbox=(10, 10, 30, 50),
        center_px=(20.0, 30.0),
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
        source="p2_closure_test",
    )


class P2ClosureTest(unittest.TestCase):
    def test_same_handler_closes_loose_interval_without_fake_transition(self) -> None:
        tracker = PossessionTrackerV1(
            lock_frames=1,
            switch_frames=1,
            lost_grace_frames=6,
            loose_confirm_frames=2,
        )
        teams = {1: ("blue", 0.95)}

        tracker.update(
            frame_id=0,
            players=[player(0, 1)],
            ball_state=ball(0, 20, 30),
            team_evidence=teams,
        )
        tracker.update(
            frame_id=1,
            players=[player(1, 1)],
            ball_state=ball(1, 45, 30),
            team_evidence=teams,
        )
        loose_events = tracker.update(
            frame_id=2,
            players=[player(2, 1)],
            ball_state=ball(2, 45, 30),
            team_evidence=teams,
        )
        self.assertIn(
            BasketballEventType.LOOSE_BALL_STARTED,
            {event.event_type for event in loose_events},
        )
        self.assertEqual(tracker.control_state, BallControlState.LOOSE)
        self.assertEqual(tracker.current_track_id, 1)

        recovery_events = tracker.update(
            frame_id=3,
            players=[player(3, 1)],
            ball_state=ball(3, 20, 30),
            team_evidence=teams,
        )
        event_types = {event.event_type for event in recovery_events}
        self.assertEqual(event_types, {BasketballEventType.LOOSE_BALL_RECOVERED})
        recovered = recovery_events[0]
        self.assertEqual(recovered.actor_track_id, 1)
        self.assertEqual(recovered.secondary_track_id, 1)
        self.assertEqual(recovered.team_id, "blue")
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)
        self.assertIsNone(tracker.diagnostics()["loose_start_frame"])

    def test_detection_only_is_not_a_canonical_analyzer_or_probe_contract(self) -> None:
        analyzer_path = PROJECT_ROOT / "services" / "analytics" / "analyzer.py"
        analyzer_source = analyzer_path.read_text(encoding="utf-8")
        analyzer_tree = ast.parse(analyzer_source)
        main_node = next(
            node
            for node in analyzer_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        arg_names = [arg.arg for arg in main_node.args.args]
        self.assertNotIn("detection_only", arg_names)
        self.assertNotIn("--detection-only", analyzer_source)
        self.assertIn('config.pop("detection_only", None)', analyzer_source)

        probe_source = (PROJECT_ROOT / "evaluation" / "p2_real_video_probe.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("detection_only=True", probe_source)
        self.assertNotIn("detection_only=False", probe_source)


if __name__ == "__main__":
    unittest.main()
