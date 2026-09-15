from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import BasketballEventType, ShotOutcome
from core.court_state import CourtState
from core.player_state import PlayerState
from services.p2.possession_tracker_v1 import PossessionTrackerV1
from services.p2.shot_event_tracker_v1 import ShotEventTrackerV1


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


class PossessionTrackerV1Test(unittest.TestCase):
    def test_same_team_switch_emits_pass_candidate(self) -> None:
        tracker = PossessionTrackerV1(lock_frames=2, switch_frames=2)
        teams = {1: ("blue", 0.95), 2: ("blue", 0.95)}

        events = []
        for frame_id in (0, 1):
            frame_players = [
                player(frame_id, 1, (10, 10, 30, 50)),
                player(frame_id, 2, (50, 10, 70, 50)),
            ]
            events.extend(
                tracker.update(
                    frame_id=frame_id,
                    players=frame_players,
                    ball_state=ball(frame_id, 20, 30),
                    team_evidence=teams,
                )
            )

        self.assertIn(BasketballEventType.POSSESSION_ACQUIRED, {event.event_type for event in events})

        switch_events = []
        for frame_id in (2, 3):
            frame_players = [
                player(frame_id, 1, (10, 10, 30, 50)),
                player(frame_id, 2, (50, 10, 70, 50)),
            ]
            switch_events.extend(
                tracker.update(
                    frame_id=frame_id,
                    players=frame_players,
                    ball_state=ball(frame_id, 60, 30),
                    team_evidence=teams,
                )
            )

        event_types = {event.event_type for event in switch_events}
        self.assertIn(BasketballEventType.POSSESSION_CHANGE, event_types)
        self.assertIn(BasketballEventType.PASS_CANDIDATE, event_types)
        self.assertNotIn(BasketballEventType.TURNOVER_CANDIDATE, event_types)

    def test_unknown_team_does_not_force_pass_or_turnover(self) -> None:
        tracker = PossessionTrackerV1(lock_frames=1, switch_frames=1)
        p1 = player(0, 1, (10, 10, 30, 50))
        p2 = player(1, 2, (50, 10, 70, 50))
        tracker.update(frame_id=0, players=[p1], ball_state=ball(0, 20, 30))
        events = tracker.update(frame_id=1, players=[p2], ball_state=ball(1, 60, 30))
        event_types = {event.event_type for event in events}
        self.assertIn(BasketballEventType.POSSESSION_CHANGE, event_types)
        self.assertNotIn(BasketballEventType.PASS_CANDIDATE, event_types)
        self.assertNotIn(BasketballEventType.TURNOVER_CANDIDATE, event_types)


class ShotEventTrackerV1Test(unittest.TestCase):
    def test_observed_downward_rim_crossing_resolves_made(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        self.assertEqual(
            tracker.update(
                frame_id=0,
                timestamp_sec=0.0,
                ball_state=ball(0, 100, 70),
                court_state=court(0),
            ),
            [],
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 100, 90),
            court_state=court(1),
        )
        events = tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 100, 100),
            court_state=court(2),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, BasketballEventType.SHOT_ATTEMPT)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.MADE.value)

    def test_observed_exit_outside_rim_resolves_missed(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 100, 70),
            court_state=court(0),
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 100, 90),
            court_state=court(1),
        )
        events = tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 140, 103),
            court_state=court(2),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.MISSED.value)

    def test_ball_in_basket_semantic_alone_cannot_create_shot(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        events = tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 300, 300),
            court_state=court(0),
            semantic_evidence=[
                {"class_name": "ball-in-basket", "confidence": 0.99, "box": [90, 90, 110, 110]}
            ],
        )
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
