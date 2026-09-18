from __future__ import annotations

import unittest

from core.ball_state import BallState, BallStateStatus
from core.court_state import CourtState
from services.p2.shot_event_tracker_v1 import ShotEventTrackerV1


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


class ShotStartGateGeometryDiagnosticsTest(unittest.TestCase):
    def test_normalized_offsets_and_consecutive_vertical_delta(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)

        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 104.0, 90.0),
            court_state=court(0),
        )
        first = tracker.diagnostics()["start_gate_diagnostics"]
        self.assertAlmostEqual(first["x_offset_rim_widths"], 0.2)
        self.assertAlmostEqual(first["y_offset_rim_heights"], -0.5)
        self.assertIsNone(first["delta_y_offset_rim_heights"])

        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 104.0, 100.0),
            court_state=court(1),
        )
        second = tracker.diagnostics()["start_gate_diagnostics"]
        self.assertTrue(second["geometry_consecutive"])
        self.assertTrue(second["rim_geometry_coherent_with_previous"])
        self.assertAlmostEqual(second["y_offset_rim_heights"], 0.5)
        self.assertAlmostEqual(second["delta_y_offset_rim_heights"], 1.0)

    def test_reset_breaks_geometry_chain(self) -> None:
        tracker = ShotEventTrackerV1(fps=10.0)
        tracker.update(
            frame_id=0,
            timestamp_sec=0.0,
            ball_state=ball(0, 100.0, 90.0),
            court_state=court(0),
        )
        tracker.reset(reason="test_reset")
        tracker.update(
            frame_id=1,
            timestamp_sec=0.1,
            ball_state=ball(1, 100.0, 100.0),
            court_state=court(1),
        )
        diagnostics = tracker.diagnostics()["start_gate_diagnostics"]
        self.assertFalse(diagnostics["geometry_consecutive"])
        self.assertFalse(diagnostics["rim_geometry_coherent_with_previous"])
        self.assertIsNone(diagnostics["delta_y_offset_rim_heights"])


if __name__ == "__main__":
    unittest.main()
