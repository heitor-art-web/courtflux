from __future__ import annotations

import unittest

from evaluation.p2_gate_b_scoreboard import (
    CHALLENGE_CLIPS,
    DEVELOPMENT_CLIPS,
    _aggregate,
    _gate_checks,
    _same_gt_window_duplicates,
)


class GateBScoreboardTest(unittest.TestCase):
    def test_dataset_split_keeps_v2_non_blocking(self) -> None:
        self.assertEqual(DEVELOPMENT_CLIPS, ("V1", "V3", "V4", "V5"))
        self.assertEqual(CHALLENGE_CLIPS, ("V2",))

    def test_duplicate_detector_counts_extra_prediction_for_same_gt_window(self) -> None:
        truth = [{"event_type": "shot_attempt", "frame_start": 100, "frame_end": 110}]
        predicted = [
            {"event_type": "shot_attempt", "frame_start": 102, "frame_end": 102},
            {"event_type": "shot_attempt", "frame_start": 108, "frame_end": 108},
        ]
        result = _same_gt_window_duplicates(predicted, truth, tolerance_frames=8)
        self.assertEqual(result["duplicate_predictions"], 1)
        self.assertEqual(result["duplicate_gt_windows"], 1)

    def test_gate_passes_frozen_thresholds(self) -> None:
        clips = [
            {
                "ground_truth_shots": 5,
                "predicted_shots": 5,
                "shot": {
                    "tp": 4,
                    "fp": 1,
                    "fn": 1,
                    "outcome_labeled_matches": 4,
                    "outcome_correct": 3,
                },
                "duplicates": {"duplicate_predictions": 0, "duplicate_gt_windows": 0},
            }
        ]
        aggregate = _aggregate(clips)
        gate = _gate_checks(aggregate)
        self.assertEqual(aggregate["precision"], 0.8)
        self.assertEqual(aggregate["recall"], 0.8)
        self.assertEqual(aggregate["outcome_accuracy"], 0.75)
        self.assertEqual(gate["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
