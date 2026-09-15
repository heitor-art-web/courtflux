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
        self.assertEqual(result["by_type"]["pass_candidate"]["fp"], 1)
        self.assertEqual(result["by_type"]["turnover_candidate"]["fn"], 1)

    def test_control_state_intervals_evaluate_only_labeled_frames(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
