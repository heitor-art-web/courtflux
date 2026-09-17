from __future__ import annotations

import unittest
from types import SimpleNamespace

from services.p2.metrics import P2MetricsCollector


class P2SoftAcquisitionRuntimeTest(unittest.TestCase):
    @staticmethod
    def frame(
        frame_id: int,
        *,
        raw_track_id: int | None,
        raw_score: float,
        control_state: str = "unknown",
        current_track_id: int | None = None,
        candidate_track_id: int | None = None,
        flight_active: bool = False,
        shot_active: bool = False,
        engine_eligible: bool = True,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            frame_id=frame_id,
            timestamp=frame_id / 25.0,
            fps=25.0,
            valid_gameplay=True,
            court_state=None,
            diagnostics={
                "p2": {
                    "p1_state_is_not_mutated": True,
                    "engine_eligible": engine_eligible,
                    "possession": {
                        "control_state": control_state,
                        "current_track_id": current_track_id,
                        "current_team_id": None,
                        "current_confidence": 0.0,
                        "lost_frames": 0,
                        "candidate_track_id": candidate_track_id,
                        "candidate_frames": 0 if candidate_track_id is None else 1,
                        "candidate_score": 0.0,
                        "candidate_peak_score": 0.0,
                        "candidate_threshold": 0.58,
                        "candidate_required_frames": 3,
                        "raw_best_track_id": raw_track_id,
                        "raw_best_score": raw_score,
                        "raw_best_above_threshold": raw_score >= 0.58,
                        "raw_best_distinct_from_flight_origin": False,
                        "pass_flight_receiver_evidence": False,
                        "flight_classification_policy": "test",
                        "flight": {
                            "active": flight_active,
                            "origin_track_id": 7 if flight_active else None,
                            "flight_supported": flight_active,
                            "flight_frames": 1 if flight_active else 0,
                            "flight_direct_observations": 1 if flight_active else 0,
                            "flight_predicted_observations": 0,
                            "flight_displacement_px": 100.0 if flight_active else 0.0,
                            "flight_displacement_player_heights": 1.0 if flight_active else 0.0,
                        },
                    },
                    "shot": {
                        "active": shot_active,
                        "awaiting_rearm": False,
                        "rearm_clear_observed_frames": 0,
                    },
                    "temporal_reset": {"count": 0, "last_reason": None},
                    "events": [],
                }
            },
        )

    def test_four_frame_soft_candidate_triggers_without_mutating_live_state(self) -> None:
        collector = P2MetricsCollector()
        for frame_id, score in enumerate((0.37, 0.36, 0.35, 0.34), start=10):
            collector.observe(self.frame(frame_id, raw_track_id=13, raw_score=score))

        rows = {int(row["frame_id"]): row for row in collector.state_rows}
        self.assertFalse(rows[12]["soft_acquisition_triggered"])
        self.assertTrue(rows[13]["soft_acquisition_triggered"])
        self.assertEqual(rows[13]["soft_acquisition_track_id"], 13)
        self.assertEqual(rows[13]["soft_acquisition_streak_frames"], 4)
        self.assertEqual(rows[13]["soft_acquisition_challenger_state"], "possessed")
        self.assertEqual(rows[13]["control_state"], "unknown")
        self.assertEqual(rows[13]["live_control_state"], "unknown")

    def test_below_mean_threshold_does_not_trigger(self) -> None:
        collector = P2MetricsCollector()
        for frame_id, score in enumerate((0.30, 0.31, 0.32, 0.33), start=20):
            collector.observe(self.frame(frame_id, raw_track_id=8, raw_score=score))

        row = collector.state_rows[-1]
        self.assertEqual(row["soft_acquisition_streak_frames"], 4)
        self.assertFalse(row["soft_acquisition_triggered"])
        self.assertEqual(row["soft_acquisition_challenger_state"], "unknown")

    def test_flight_or_thresholded_candidate_resets_soft_streak(self) -> None:
        collector = P2MetricsCollector()
        collector.observe(self.frame(30, raw_track_id=5, raw_score=0.40))
        collector.observe(self.frame(31, raw_track_id=5, raw_score=0.40))
        collector.observe(
            self.frame(32, raw_track_id=5, raw_score=0.40, flight_active=True)
        )
        collector.observe(self.frame(33, raw_track_id=5, raw_score=0.40))
        self.assertEqual(collector.state_rows[-1]["soft_acquisition_streak_frames"], 1)

        collector.observe(
            self.frame(34, raw_track_id=5, raw_score=0.60, candidate_track_id=5)
        )
        self.assertFalse(collector.state_rows[-1]["soft_acquisition_eligible"])
        collector.observe(self.frame(35, raw_track_id=5, raw_score=0.40))
        self.assertEqual(collector.state_rows[-1]["soft_acquisition_streak_frames"], 1)


if __name__ == "__main__":
    unittest.main()
