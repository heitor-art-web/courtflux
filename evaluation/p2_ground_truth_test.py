from __future__ import annotations

import unittest

from evaluation.p2_ground_truth import evaluate_control_states, evaluate_events


class P2GroundTruthEvaluatorTest(unittest.TestCase):
    def test_event_matching_respects_type_outcome_and_tolerance(self) -> None:
        predicted = [
            {
                "event_type": "shot_attempt",
                "frame_end": 100,
                "attributes": {"outcome": "made"},
            },
            {
                "event_type": "pass_candidate",
                "frame_end": 150,
                "attributes": {},
            },
        ]
        truth = [
            {"event_type": "shot_attempt", "frame": 103, "outcome": "made"},
            {"event_type": "turnover_candidate", "frame": 170},
        ]
        result = evaluate_events(predicted, truth, tolerance_frames=5)
        self.assertEqual(result["matched_events"], 1)
        self.assertEqual(result["by_type"]["shot_attempt"]["tp"], 1)
        self.assertEqual(result["by_type"]["shot_attempt"]["f1"], 1.0)
        self.assertEqual(result["by_type"]["shot_attempt"]["mean_abs_frame_error"], 3.0)
        self.assertEqual(result["by_type"]["pass_candidate"]["fp"], 1)
        self.assertEqual(result["by_type"]["turnover_candidate"]["fn"], 1)

    def test_event_inside_human_window_has_zero_temporal_error(self) -> None:
        predicted = [
            {
                "event_type": "shot_attempt",
                "frame_end": 118,
                "attributes": {"outcome": "made"},
            }
        ]
        truth = [
            {
                "event_type": "shot_attempt",
                "frame_start": 100,
                "frame_end": 125,
                "outcome": "made",
            }
        ]
        result = evaluate_events(predicted, truth, tolerance_frames=0, fps=25.0)
        self.assertEqual(result["matched_events"], 1)
        self.assertEqual(result["ground_truth_window_events"], 1)
        match = result["matches"][0]
        self.assertTrue(match["inside_truth_window"])
        self.assertEqual(match["frame_error"], 0)
        metrics = result["by_type"]["shot_attempt"]
        self.assertEqual(metrics["mean_distance_to_human_window_frames"], 0.0)
        self.assertEqual(metrics["mean_distance_to_human_window_sec"], 0.0)

    def test_event_tolerance_extends_beyond_window_edges(self) -> None:
        predicted = [
            {
                "event_type": "shot_attempt",
                "frame_end": 130,
                "attributes": {"outcome": "made"},
            }
        ]
        truth = [
            {
                "event_type": "shot_attempt",
                "frame_start": 100,
                "frame_end": 125,
                "outcome": "made",
            }
        ]
        result = evaluate_events(predicted, truth, tolerance_frames=5)
        self.assertEqual(result["matched_events"], 1)
        self.assertEqual(result["matches"][0]["frame_error"], 5)

    def test_control_state_intervals_evaluate_frames_and_continuity(self) -> None:
        predicted = [
            {"frame_id": 10, "control_state": "possessed"},
            {"frame_id": 11, "control_state": "possessed"},
            {"frame_id": 12, "control_state": "pass_flight"},
            {"frame_id": 13, "control_state": "loose"},
        ]
        truth = [
            {"state": "possessed", "frame_start": 10, "frame_end": 11},
            {"state": "pass_flight", "frame_start": 12, "frame_end": 13},
        ]
        result = evaluate_control_states(predicted, truth)
        self.assertEqual(result["ground_truth_labeled_frames"], 4)
        self.assertEqual(result["evaluated_frames"], 4)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion"]["pass_flight"]["loose"], 1)
        continuity = result["continuity"]
        self.assertEqual(continuity["evaluated_intervals"], 2)
        self.assertEqual(continuity["dominant_state_interval_accuracy"], 1.0)
        self.assertEqual(continuity["mean_interval_matching_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
