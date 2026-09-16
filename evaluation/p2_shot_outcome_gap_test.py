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


def court(
    frame_id: int,
    *,
    rim_visible: bool = True,
    rim_bbox: tuple[float, float, float, float] = (90.0, 90.0, 110.0, 100.0),
) -> CourtState:
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
                    "bbox_px": list(rim_bbox),
                    "center_px": [
                        (rim_bbox[0] + rim_bbox[2]) / 2.0,
                        (rim_bbox[1] + rim_bbox[3]) / 2.0,
                    ],
                    "confidence": 0.95,
                    "event_truth": False,
                }
            ]
            if rim_visible
            else []
        ),
    )


class ShotOutcomeGapRegressionTest(unittest.TestCase):
    def test_rim_plane_intersection_accepts_diagonal_crossing_with_outside_endpoints(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 70, 70),
            court_state=court(0),
        )
        events = tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 120, 105),
            court_state=court(1),
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.MADE.value)
        self.assertEqual(
            events[0].attributes["resolution_reason"],
            "observed_downward_rim_plane_crossing",
        )
        self.assertIsNotNone(events[0].attributes["crossing_x_px"])
        self.assertIsNotNone(events[0].attributes["crossing_fraction"])

    def test_gap_crossing_creates_pending_bridge_and_ratio_stays_diagnostic(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 95, 75),
            court_state=court(0),
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 98, 88),
            court_state=court(1),
        )
        tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 100, 96, status=BallStateStatus.PREDICTED),
            court_state=court(2),
        )

        bridge_frame = tracker.update(
            frame_id=3,
            timestamp_sec=0.3,
            ball_state=ball(3, 102, 102),
            court_state=court(3),
        )
        self.assertEqual(bridge_frame, [])
        self.assertTrue(tracker.diagnostics()["pending_bridge"])

        events = tracker.update(
            frame_id=4,
            timestamp_sec=0.4,
            ball_state=ball(4, 103, 107),
            court_state=court(4),
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertEqual(
            event.attributes["resolution_reason"],
            "pending_bridge_post_gap_ratio_unvalidated",
        )
        self.assertTrue(event.attributes["pending_bridge_used"])
        self.assertIsNotNone(event.attributes["post_gap_fall_ratio"])
        self.assertFalse(event.attributes["post_gap_fall_ratio_validated_for_outcome"])

    def test_short_rim_gap_can_bridge_only_with_reappearing_coherent_geometry(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 95, 75),
            court_state=court(0),
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 98, 88),
            court_state=court(1),
        )
        tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 100, 96, status=BallStateStatus.PREDICTED),
            court_state=court(2, rim_visible=False),
        )
        self.assertEqual(
            tracker.update(
                frame_id=3,
                timestamp_sec=0.3,
                ball_state=ball(3, 102, 102),
                court_state=court(3),
            ),
            [],
        )
        self.assertTrue(tracker.diagnostics()["pending_bridge"])

        events = tracker.update(
            frame_id=4,
            timestamp_sec=0.4,
            ball_state=ball(4, 103, 107),
            court_state=court(4),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertGreaterEqual(events[0].attributes["bridge_rim_gap_frames"], 1)

    def test_direct_ball_can_continue_while_rim_is_missing_until_geometry_recovers(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 95, 75),
            court_state=court(0),
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 98, 88),
            court_state=court(1),
        )
        for frame_id, y in ((2, 100), (3, 108), (4, 116)):
            tracker.update(
                frame_id=frame_id,
                timestamp_sec=frame_id / 10.0,
                ball_state=ball(frame_id, 100 + frame_id, y),
                court_state=court(frame_id, rim_visible=False),
            )

        self.assertEqual(
            tracker.update(
                frame_id=5,
                timestamp_sec=0.5,
                ball_state=ball(5, 104, 120),
                court_state=court(5),
            ),
            [],
        )
        self.assertTrue(tracker.diagnostics()["pending_bridge"])
        self.assertGreaterEqual(tracker.diagnostics()["max_rim_evidence_gap_frames"], 3)

        events = tracker.update(
            frame_id=6,
            timestamp_sec=0.6,
            ball_state=ball(6, 105, 125),
            court_state=court(6),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].attributes["outcome"], ShotOutcome.UNKNOWN.value)
        self.assertIsNotNone(events[0].attributes["post_gap_fall_ratio"])

    def test_camera_jump_does_not_bridge_non_overlapping_rim_geometry(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0, reset_time_sec=0.5)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 95, 75),
            court_state=court(0),
        )
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 98, 88),
            court_state=court(1),
        )
        tracker.update(
            frame_id=2,
            timestamp_sec=0.2,
            ball_state=ball(2, 100, 96, status=BallStateStatus.PREDICTED),
            court_state=court(2, rim_visible=False),
        )
        tracker.update(
            frame_id=3,
            timestamp_sec=0.3,
            ball_state=ball(3, 302, 302),
            court_state=court(3, rim_bbox=(290.0, 290.0, 310.0, 300.0)),
        )
        self.assertFalse(tracker.diagnostics()["pending_bridge"])

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
