from __future__ import annotations

import unittest

from evaluation.p2_ground_truth import evaluate_control_states, evaluate_events


class P2GroundTruthEvaluatorTest(unittest.TestCase):
    def test_event_matching_scores_outcome_separately_and_ignores_unlabeled_types(self) -> None:
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
        self.assertEqual(result["evaluated_predicted_events"], 1)
        self.assertEqual(result["not_evaluated_predicted_events"], 1)
        self.assertEqual(result["not_evaluated_prediction_types"], {"pass_candidate": 1})
        self.assertNotIn("pass_candidate", result["by_type"])
        self.assertEqual(result["by_type"]["shot_attempt"]["tp"], 1)
        self.assertEqual(result["by_type"]["shot_attempt"]["f1"], 1.0)
        self.assertEqual(result["by_type"]["shot_attempt"]["outcome_accuracy"], 0.0)
        self.assertEqual(result["by_type"]["turnover_candidate"]["fn"], 1)
        self.assertEqual(result["overall"]["tp"], 1)
        self.assertEqual(result["overall"]["fp"], 0)
        self.assertEqual(result["overall"]["fn"], 1)
        self.assertEqual(result["overall"]["precision"], 1.0)
        self.assertEqual(result["overall"]["recall"], 0.5)

    def test_prediction_only_event_types_do_not_become_false_positives(self) -> None:
        predicted = [
            {
                "event_type": "shot_attempt",
                "frame_start": 100,
                "frame_end": 105,
                "attributes": {"outcome": "made"},
            },
            {"event_type": "loose_ball_started", "frame_start": 120, "frame_end": 120},
            {"event_type": "loose_ball_recovered", "frame_start": 130, "frame_end": 130},
            {"event_type": "possession_acquired", "frame_start": 140, "frame_end": 140},
        ]
        truth = [
            {
                "event_type": "shot_attempt",
                "frame_start": 100,
                "frame_end": 110,
                "outcome": "made",
            }
        ]
        result = evaluate_events(predicted, truth, tolerance_frames=8, fps=25.0)
        self.assertEqual(result["overall"]["tp"], 1)
        self.assertEqual(result["overall"]["fp"], 0)
        self.assertEqual(result["overall"]["fn"], 0)
        self.assertEqual(result["overall"]["f1"], 1.0)
        self.assertEqual(result["evaluated_predicted_events"], 1)
        self.assertEqual(result["not_evaluated_predicted_events"], 3)

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
