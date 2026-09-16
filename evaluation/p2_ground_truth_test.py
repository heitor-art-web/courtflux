from __future__ import annotations

import unittest

from evaluation.p2_ground_truth import evaluate_control_states, evaluate_events


class P2GroundTruthEvaluatorTest(unittest.TestCase):
    def test_event_matching_scores_outcome_separately(self) -> None:
        predicted = [
            {
                "event_type": "shot_attempt",
                "frame_start": 98,
                "frame_end": 105,
                "attributes": {"outcome": "missed"},
            },
            {
                "event_type": "pass_candidate",
                "frame_start": 150,
                "frame_end": 155,
                "attributes": {},
            },
        ]
        truth = [
            {
                "event_type": "shot_attempt",
                "frame_start": 100,
                "frame_end": 110,
                "outcome": "made",
            },
            {"event_type": "turnover_candidate", "frame": 170},
        ]
        result = evaluate_events(predicted, truth, tolerance_frames=5, fps=25.0)
        self.assertEqual(result["matched_events"], 1)
        self.assertEqual(result["by_type"]["shot_attempt"]["tp"], 1)
        self.assertEqual(result["by_type"]["shot_attempt"]["f1"], 1.0)
        self.assertEqual(result["by_type"]["shot_attempt"]["outcome_accuracy"], 0.0)
        self.assertEqual(result["by_type"]["pass_candidate"]["fp"], 1)
        self.assertEqual(result["by_type"]["turnover_candidate"]["fn"], 1)

    def test_event_windows_use_edge_distance_not_frame_end(self) -> None:
        predicted = [
            {
                "event_type": "shot_attempt",
                "frame_start": 276,
                "frame_end": 318,
                "attributes": {"outcome": "made"},
            }
        ]
        truth = [
            {
                "event_type": "shot_attempt",
                "frame_start": 225,
                "frame_end": 275,
                "outcome": "made",
            }
        ]
        result = evaluate_events(predicted, truth, tolerance_frames=8, fps=25.0)
        self.assertEqual(result["matched_events"], 1)
        match = result["matches"][0]
        self.assertEqual(match["frame_distance_between_windows"], 1)
        self.assertFalse(match["windows_overlap"])
        self.assertEqual(result["by_type"]["shot_attempt"]["outcome_accuracy"], 1.0)

    def test_control_state_intervals_evaluate_continuity(self) -> None:
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
        self.assertEqual(result["continuity"]["evaluated_intervals"], 2)
        self.assertEqual(result["continuity"]["mean_interval_matching_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
