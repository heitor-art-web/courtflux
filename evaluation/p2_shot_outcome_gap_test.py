from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from core.basketball_event import ShotOutcome
from core.court_state import CourtState
from services.p2.shot_event_tracker_v1 import ShotEventTrackerV1


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


def court(frame_id: int, *, rim_visible: bool = True) -> CourtState:
    return CourtState(
        frame_id=frame_id,
        timestamp_sec=frame_id / 10.0,
        mode="image_space_fallback",
        court_visible=True,
        analysis_eligible=True,
        basket_evidence=(
            [
                {
                    "kind": "rim",
                    "bbox_px": [90.0, 90.0, 110.0, 100.0],
                    "center_px": [100.0, 95.0],
                    "confidence": 0.95,
                    "event_truth": False,
                }
            ]
            if rim_visible
            else []
        ),
    )


class ShotOutcomeGapRegressionTest(unittest.TestCase):
    def test_bracketed_direct_observations_can_resolve_made_across_predicted_hold(self) -> None:
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
            ball_state=ball(1, 100, 100, status=BallStateStatus.PREDICTED),
            court_state=court(1),
        )
        events = tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 100, 102),
            court_state=court(2),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.MADE.value)
        self.assertEqual(
            events[0].attributes["resolution_reason"],
            "observed_bracketed_rim_crossing_after_gap",
        )
        self.assertEqual(events[0].attributes["max_ball_observation_gap_frames"], 1)

    def test_outside_reappearance_after_ball_gap_abstains_instead_of_missed(self) -> None:
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
        tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 120, 100, status=BallStateStatus.PREDICTED),
            court_state=court(2),
        )
        events = tracker.update(
            frame_id=3,
            timestamp_sec=0.3,
            ball_state=ball(3, 140, 104),
            court_state=court(3),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertEqual(
            events[0].attributes["resolution_reason"],
            "rim_exit_after_observation_gap_insufficient_for_miss",
        )
        self.assertTrue(events[0].attributes["outcome_geometry_tainted"])

    def test_rim_gap_taints_later_outside_exit_even_after_direct_ball_recovery(self) -> None:
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
        tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 100, 100),
            court_state=court(2, rim_visible=False),
        )
        tracker.update(
            frame_id=3,
            timestamp_sec=0.3,
            ball_state=ball(3, 105, 103),
            court_state=court(3),
        )
        events = tracker.update(
            frame_id=4,
            timestamp_sec=0.4,
            ball_state=ball(4, 140, 104),
            court_state=court(4),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertEqual(
            events[0].attributes["resolution_reason"],
            "rim_exit_after_observation_gap_insufficient_for_miss",
        )
        self.assertGreaterEqual(events[0].attributes["max_rim_evidence_gap_frames"], 1)

    def test_missing_rim_never_reuses_stale_bbox_to_resolve_missed(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0, reset_time_sec=0.3)
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
        self.assertEqual(
            tracker.update(
                frame_id=2,
                timestamp_sec=0.2,
                ball_state=ball(2, 140, 104),
                court_state=court(2, rim_visible=False),
            ),
            [],
        )
        events = tracker.update(
            frame_id=3,
            timestamp_sec=0.3,
            ball_state=ball(3, 140, 106),
            court_state=court(3, rim_visible=False),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertEqual(
            events[0].attributes["resolution_reason"],
            "shot_window_expired_without_outcome_evidence",
        )


if __name__ == "__main__":
    unittest.main()
