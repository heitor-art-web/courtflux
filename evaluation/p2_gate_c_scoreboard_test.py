from __future__ import annotations

import unittest

from evaluation.p2_gate_c_scoreboard import (
    _aggregate,
    _evaluate_event_sample,
    _evaluate_possessed_sample,
    _gate,
)


class GateCScoreboardTest(unittest.TestCase):
    def test_possessed_modal_team_separates_accuracy_and_contamination(self) -> None:
        sample = {
            "state": "possessed",
            "frame_start": 10,
            "frame_end": 13,
            "team_id": "red",
        }
        states = [
            {"frame_id": 10, "current_team_id": "red"},
            {"frame_id": 11, "current_team_id": "red"},
            {"frame_id": 12, "current_team_id": "blue"},
            {"frame_id": 13, "current_team_id": None},
        ]
        result = _evaluate_possessed_sample(sample, states)
        self.assertTrue(result["attributed"])
        self.assertTrue(result["correct"])
        self.assertEqual(result["predicted_team_id"], "red")
        self.assertEqual(result["evidence_points"], 3)
        self.assertEqual(result["contamination_rate"], 0.3333)

    def test_unknown_possessed_team_is_unattributed_not_wrong(self) -> None:
        sample = {
            "state": "possessed",
            "frame_start": 1,
            "frame_end": 2,
            "team_id": "white",
        }
        states = [
            {"frame_id": 1, "current_team_id": None},
            {"frame_id": 2, "current_team_id": "unknown"},
        ]
        result = _evaluate_possessed_sample(sample, states)
        self.assertFalse(result["attributed"])
        self.assertFalse(result["correct"])
        self.assertIsNone(result["predicted_team_id"])
        self.assertIsNone(result["contamination_rate"])

    def test_event_matching_is_one_to_one(self) -> None:
        events = [
            {
                "event_type": "shot_attempt",
                "frame_start": 20,
                "frame_end": 25,
                "team_id": "green",
            }
        ]
        first = {
            "state": "shot_flight",
            "frame_start": 18,
            "frame_end": 26,
            "team_id": "green",
        }
        second = {
            "state": "shot_flight",
            "frame_start": 24,
            "frame_end": 30,
            "team_id": "green",
        }
        used: set[int] = set()
        first_result = _evaluate_event_sample(
            first,
            events,
            tolerance_frames=8,
            used_event_indices=used,
        )
        second_result = _evaluate_event_sample(
            second,
            events,
            tolerance_frames=8,
            used_event_indices=used,
        )
        self.assertTrue(first_result["attributed"])
        self.assertTrue(first_result["correct"])
        self.assertFalse(second_result["attributed"])
        self.assertEqual(used, {0})

    def test_gate_uses_frozen_thresholds(self) -> None:
        clips = [
            {
                "clip": "V1",
                "samples": 5,
                "attributed_samples": 5,
                "correct_samples": 5,
                "coverage": 1.0,
                "accuracy": 1.0,
                "sample_results": [
                    {"contamination_rate": 0.0} for _ in range(5)
                ],
            },
            {
                "clip": "V3",
                "samples": 5,
                "attributed_samples": 4,
                "correct_samples": 4,
                "coverage": 0.8,
                "accuracy": 1.0,
                "sample_results": [
                    {"contamination_rate": 0.0} for _ in range(4)
                ] + [{"contamination_rate": None}],
            },
        ]
        aggregate = _aggregate(clips)
        gate = _gate(aggregate, clips)
        self.assertEqual(aggregate["coverage"], 0.9)
        self.assertEqual(aggregate["accuracy"], 1.0)
        self.assertEqual(aggregate["cross_team_contamination"], 0.0)
        self.assertEqual(gate["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
