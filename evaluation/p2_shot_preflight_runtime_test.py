from __future__ import annotations

import unittest
from types import SimpleNamespace

from services.p2.metrics import P2MetricsCollector


class _Ball:
    def __init__(self, x: float, y: float) -> None:
        self.available = True
        self.status = SimpleNamespace(name="OBSERVED")
        self._position = (x, y)

    def position(self):
        return self._position


class _Court:
    def __init__(self) -> None:
        self.basket_evidence = [
            {"kind": "rim", "bbox_px": [90.0, 90.0, 110.0, 100.0]}
        ]


class P2ShotPreflightRuntimeTest(unittest.TestCase):
    @staticmethod
    def frame(frame_id: int, *, ball_y: float) -> SimpleNamespace:
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=frame_id / 25.0,
            fps=25.0,
            valid_gameplay=True,
            court_state=_Court(),
            ball_state=_Ball(100.0, ball_y),
            diagnostics={
                "p2": {
                    "p1_state_is_not_mutated": True,
                    "engine_eligible": True,
                    "possession": {
                        "control_state": "unknown",
                        "current_track_id": None,
                        "current_team_id": None,
                        "current_confidence": 0.0,
                        "lost_frames": 0,
                        "candidate_track_id": None,
                        "candidate_frames": 0,
                        "candidate_score": 0.0,
                        "candidate_peak_score": 0.0,
                        "candidate_threshold": 0.58,
                        "candidate_required_frames": 3,
                        "raw_best_track_id": None,
                        "raw_best_score": 0.0,
                        "raw_best_above_threshold": False,
                        "raw_best_distinct_from_flight_origin": False,
                        "pass_flight_receiver_evidence": False,
                        "flight_classification_policy": "test",
                        "flight": {
                            "active": True,
                            "origin_track_id": 7,
                            "flight_supported": True,
                            "flight_frames": frame_id,
                            "flight_direct_observations": frame_id,
                            "flight_predicted_observations": 0,
                            "flight_displacement_px": 50.0,
                            "flight_displacement_player_heights": 1.0,
                        },
                    },
                    "shot": {
                        "active": False,
                        "awaiting_rearm": False,
                        "rearm_clear_observed_frames": 0,
                    },
                    "temporal_reset": {"count": 0, "last_reason": None},
                    "events": [],
                }
            },
        )

    def test_preflight_tracks_causal_rising_and_approach_without_mutating_live_state(self) -> None:
        collector = P2MetricsCollector()
        collector.observe(self.frame(1, ball_y=130.0))
        collector.observe(self.frame(2, ball_y=120.0))
        collector.observe(self.frame(3, ball_y=110.0))

        rows = {int(row["frame_id"]): row for row in collector.state_rows}
        last = rows[3]
        self.assertTrue(last["shot_preflight_available"])
        self.assertTrue(last["shot_preflight_direct_observed"])
        self.assertTrue(last["shot_preflight_near_x"])
        self.assertGreaterEqual(last["shot_preflight_approach_streak"], 2)
        self.assertGreaterEqual(last["shot_preflight_rising_streak"], 2)
        self.assertLess(last["shot_preflight_delta_distance_norm"], 0.0)
        self.assertLess(last["shot_preflight_delta_relative_y"], 0.0)
        self.assertEqual(last["control_state"], "unknown")
        self.assertEqual(last["live_control_state"], "unknown")

    def test_missing_geometry_is_safe(self) -> None:
        collector = P2MetricsCollector()
        frame = self.frame(1, ball_y=120.0)
        frame.ball_state = None
        frame.court_state = None
        collector.observe(frame)
        row = collector.state_rows[-1]
        self.assertFalse(row["shot_preflight_available"])
        self.assertEqual(row["live_control_state"], "unknown")


if __name__ == "__main__":
    unittest.main()
