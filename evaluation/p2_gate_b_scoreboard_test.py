from __future__ import annotations

import unittest

from evaluation.p2_gate_b_scoreboard import (
    CHALLENGE_CLIPS,
    DEVELOPMENT_CLIPS,
    _aggregate,
    _event_gt_readiness,
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

    def test_event_gt_readiness_flags_human_shot_evidence_without_events(self) -> None:
        gt = {
            "control_intervals": [
                {"state": "shot_flight", "frame_start": 10, "frame_end": 20}
            ],
            "human_source_notes": ["made dunk"],
        }
        readiness = _event_gt_readiness(gt, [])
        self.assertFalse(readiness["complete_for_gate_b"])

    def test_gate_passes_frozen_thresholds_when_gt_is_ready(self) -> None:
        clips = [
            {
                "clip": "V1",
                "ground_truth_shots": 5,
                "predicted_shots": 5,
                "event_gt": {"complete_for_gate_b": True},
                "shot": {
                    "tp": 4,
                    "fp": 1,
                    "fn": 1,
                    "outcome_labeled_matches": 4,
                    "outcome_correct": 3,
                },
                "outcome_diagnostics": {
                    "predicted": {"made": 3, "unknown": 1},
                    "pairs": {"made->made": 3, "made->unknown": 1},
                },
                "duplicates": {"duplicate_predictions": 0, "duplicate_gt_windows": 0},
            }
        ]
        aggregate = _aggregate(clips)
        gate = _gate_checks(aggregate, clips)
        self.assertEqual(aggregate["precision"], 0.8)
        self.assertEqual(aggregate["recall"], 0.8)
        self.assertEqual(aggregate["outcome_accuracy"], 0.75)
        self.assertEqual(gate["status"], "PASS")

    def test_incomplete_gt_blocks_gate(self) -> None:
        clips = [
            {
                "clip": "V3",
                "event_gt": {"complete_for_gate_b": False},
                "ground_truth_shots": 0,
                "predicted_shots": 2,
                "shot": {},
                "outcome_diagnostics": {"predicted": {}, "pairs": {}},
                "duplicates": {"duplicate_predictions": 0, "duplicate_gt_windows": 0},
            }
        ]
        aggregate = _aggregate(clips)
        gate = _gate_checks(aggregate, clips)
        readiness_check = gate["checks"][0]
        self.assertFalse(readiness_check["passed"])
        self.assertEqual(readiness_check["actual"], ["V3"])


if __name__ == "__main__":
    unittest.main()
