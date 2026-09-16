from __future__ import annotations

import unittest
from types import SimpleNamespace

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import BallControlState, BasketballEventType, ShotOutcome
from core.court_state import CourtState
from core.player_state import PlayerState
from services.p2.engine_v1 import P2EngineV1
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


def court(frame_id: int, *, eligible: bool = True) -> CourtState:
    return CourtState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        mode="image_space_fallback",
        court_visible=eligible,
        analysis_eligible=eligible,
        basket_evidence=[
            {
                "kind": "rim",
                "bbox_px": [90.0, 90.0, 110.0, 100.0],
                "center_px": [100.0, 95.0],
                "confidence": 0.95,
                "event_truth": False,
            }
        ] if eligible else [],
    )


def frame_state(
    frame_id: int,
    *,
    eligible: bool = True,
    player_states: list[PlayerState] | None = None,
    ball_state: BallState | None = None,
    teams: dict[int, tuple[str, float]] | None = None,
):
    legacy_players = []
    for item in list(player_states or []):
        team_id, confidence = (teams or {}).get(item.track_id, (None, 0.0))
        legacy_players.append(
            {
                "track_id": item.track_id,
                "team": team_id,
                "team_confidence": confidence,
            }
        )
    return SimpleNamespace(
        frame_id=frame_id,
        timestamp=frame_id / 10.0,
        fps=10.0,
        valid_gameplay=eligible,
        player_states=list(player_states or []),
        ball_state=ball_state,
        court_state=court(frame_id, eligible=eligible),
        players=legacy_players,
        diagnostics={},
        p2_events=[],
    )


class PossessionTrackerV1Test(unittest.TestCase):
    def test_same_team_switch_with_flight_emits_pass_candidate(self) -> None:
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

        self.assertIn(
            BasketballEventType.POSSESSION_ACQUIRED,
            {event.event_type for event in events},
        )

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
        pass_event = next(
            event
            for event in switch_events
            if event.event_type is BasketballEventType.PASS_CANDIDATE
        )
        self.assertEqual(pass_event.actor_track_id, 1)
        self.assertEqual(pass_event.secondary_track_id, 2)
        self.assertTrue(pass_event.attributes["flight_supported"])

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

    def test_loose_ball_interval_and_recovery_are_explicit(self) -> None:
        tracker = PossessionTrackerV1(
            lock_frames=1,
            switch_frames=1,
            lost_grace_frames=1,
            loose_confirm_frames=2,
        )
        teams = {1: ("blue", 0.95), 2: ("red", 0.95)}
        tracker.update(
            frame_id=0,
            players=[player(0, 1, (10, 10, 30, 50))],
            ball_state=ball(0, 20, 30),
            team_evidence=teams,
        )
        # 25 px from the last controlled ball position: far enough from player 1 to
        # fail possession, but below the 0.65-player-height pass-flight displacement gate.
        tracker.update(
            frame_id=1,
            players=[player(1, 1, (10, 10, 30, 50))],
            ball_state=ball(1, 45, 30),
            team_evidence=teams,
        )
        loose_events = tracker.update(
            frame_id=2,
            players=[player(2, 1, (10, 10, 30, 50))],
            ball_state=ball(2, 45, 30),
            team_evidence=teams,
        )
        self.assertIn(
            BasketballEventType.LOOSE_BALL_STARTED,
            {event.event_type for event in loose_events},
        )
        self.assertEqual(tracker.control_state, BallControlState.LOOSE)

        recovery_events = tracker.update(
            frame_id=3,
            players=[player(3, 2, (40, 10, 50, 50))],
            ball_state=ball(3, 45, 30),
            team_evidence=teams,
        )
        recovery_types = {event.event_type for event in recovery_events}
        self.assertIn(BasketballEventType.POSSESSION_ACQUIRED, recovery_types)
        self.assertIn(BasketballEventType.LOOSE_BALL_RECOVERED, recovery_types)
        self.assertIn(BasketballEventType.TURNOVER_CANDIDATE, recovery_types)
        self.assertEqual(tracker.control_state, BallControlState.POSSESSED)


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
                {
                    "class_name": "ball-in-basket",
                    "confidence": 0.99,
                    "box": [90, 90, 110, 110],
                }
            ],
        )
        self.assertEqual(events, [])

    def test_unknown_timeout_requires_multiple_direct_clear_frames_before_rearm(self) -> None:
        tracker = ShotEventTrackerV1(
            fps=10.0,
            reset_time_sec=0.2,
            minimum_between_starts_sec=0.1,
            rearm_clear_observed_frames=4,
            rearm_min_separation_sec=0.2,
        )
        above = lambda frame_id: ball(frame_id, 100, 70)

        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=above(0),
            court_state=court(0),
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=above(1),
            court_state=court(1),
        )
        expired = tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=above(2),
            court_state=court(2),
        )
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertTrue(tracker.diagnostics()["awaiting_rearm"])

        tracker.update(
            frame_id=3,
            timestamp_sec=0.3,
            ball_state=above(3),
            court_state=court(3),
        )
        self.assertTrue(tracker.diagnostics()["awaiting_rearm"])

        # Predicted/missing-style evidence does not count as direct spatial clearance.
        tracker.update(
            frame_id=4,
            timestamp_sec=0.4,
            ball_state=ball(4, 300, 300, status=BallStateStatus.PREDICTED),
            court_state=court(4),
        )
        self.assertEqual(tracker.diagnostics()["rearm_clear_observed_frames"], 0)

        for frame_id in (5, 6, 7, 8):
            tracker.update(
                frame_id=frame_id,
                timestamp_sec=frame_id / 10.0,
                ball_state=ball(frame_id, 300, 300),
                court_state=court(frame_id),
            )
        self.assertFalse(tracker.diagnostics()["awaiting_rearm"])
        self.assertFalse(tracker.diagnostics()["active"])

        tracker.update(
            frame_id=9,
            timestamp_sec=0.9,
            ball_state=above(9),
            court_state=court(9),
        )
        self.assertTrue(tracker.diagnostics()["active"])


class P2EngineV1Test(unittest.TestCase):
    def test_shot_uses_pre_release_track_as_actor(self) -> None:
        engine = P2EngineV1(fps=10.0)
        teams = {1: ("blue", 0.95)}
        for frame_id in (0, 1, 2):
            state = frame_state(
                frame_id,
                player_states=[player(frame_id, 1, (10, 10, 30, 50))],
                ball_state=ball(frame_id, 20, 30),
                teams=teams,
            )
            engine.process(state)

        self.assertEqual(engine.possession_tracker.current_track_id, 1)
        engine.process(
            frame_state(
                3,
                player_states=[player(3, 1, (10, 10, 30, 50))],
                ball_state=ball(3, 100, 70),
                teams=teams,
            )
        )
        engine.process(
            frame_state(
                4,
                player_states=[player(4, 1, (10, 10, 30, 50))],
                ball_state=ball(4, 100, 90),
                teams=teams,
            )
        )
        resolved = frame_state(
            5,
            player_states=[player(5, 1, (10, 10, 30, 50))],
            ball_state=ball(5, 100, 100),
            teams=teams,
        )
        engine.process(resolved)
        shot = next(
            event
            for event in resolved.p2_events
            if event.event_type is BasketballEventType.SHOT_ATTEMPT
        )
        self.assertEqual(shot.actor_track_id, 1)
        self.assertEqual(shot.team_id, "blue")
        self.assertEqual(shot.attributes["actor_attribution_source"], "pre_release_possession")

    def test_ineligible_gap_resets_temporal_state(self) -> None:
        engine = P2EngineV1(fps=10.0, reset_after_ineligible_frames=2)
        teams = {1: ("blue", 0.95)}
        for frame_id in (0, 1, 2):
            engine.process(
                frame_state(
                    frame_id,
                    player_states=[player(frame_id, 1, (10, 10, 30, 50))],
                    ball_state=ball(frame_id, 20, 30),
                    teams=teams,
                )
            )
        self.assertEqual(engine.possession_tracker.current_track_id, 1)

        engine.process(frame_state(3, eligible=False))
        second_gap = frame_state(4, eligible=False)
        engine.process(second_gap)
        self.assertIsNone(engine.possession_tracker.current_track_id)
        self.assertEqual(
            second_gap.diagnostics["p2"]["temporal_reset"]["last_reason"],
            "gameplay_absent_gap",
        )
        self.assertFalse(second_gap.diagnostics["p2"]["engine_eligible"])


if __name__ == "__main__":
    unittest.main()
