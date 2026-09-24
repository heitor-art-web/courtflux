from __future__ import annotations

import unittest

from evaluation.p2_v6_validation_score import match_shots, score_gate_c, score_transition_constraints


class V6ValidationScoreTest(unittest.TestCase):
    def test_ordered_shots_keep_extra_prediction_as_false_positive_and_duplicate(self):
        gt = {
            "shot_events": [
                {"id": "S1", "block_id": "A", "window_start_sec": 0.0, "window_end_sec": 10.0, "team_id": "team_a", "outcome": None},
                {"id": "S2", "block_id": "B", "window_start_sec": 20.0, "window_end_sec": 30.0, "team_id": "team_b", "outcome": "missed", "ordered_index": 0},
                {"id": "S3", "block_id": "B", "window_start_sec": 20.0, "window_end_sec": 30.0, "team_id": "team_a", "outcome": None, "ordered_index": 1},
            ]
        }

        def shot(timestamp_sec: float, outcome: str = "unknown") -> dict[str, object]:
            return {
                "event_type": "shot_attempt",
                "timestamp_start_sec": timestamp_sec,
                "timestamp_end_sec": timestamp_sec + 0.1,
                "team_id": None,
                "attributes": {"outcome": outcome},
            }

        result = match_shots(gt, [shot(1.0), shot(2.0), shot(21.0), shot(22.0), shot(40.0)])
        self.assertEqual(result["tp"], 3)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["duplicate_predictions"], 1)

    def test_gate_c_counts_unattributed_team_labeled_shot_in_coverage(self):
        gt = {
            "video": {"end_time_sec": 10.0},
            "scoring_blocks": [
                {"id": "P1", "start_sec": 0.0, "end_sec": 5.0, "gameplay": "gameplay", "team_id": "team_a"}
            ],
            "gate_c_possession_blocks": ["P1"],
            "shot_events": [{"id": "S1", "block_id": "P1", "team_id": "team_a"}],
        }
        states = [
            {"timestamp_sec": 1.0, "control_state": "possessed", "current_team_id": "team_a"},
            {"timestamp_sec": 2.0, "control_state": "possessed", "current_team_id": "team_a"},
        ]
        gate_b = {"shots": {"matches": [{"truth_id": "S1", "matched": True, "predicted_team_id": None}]}}
        result = score_gate_c(gt, states, [], gate_b)
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["attributed_samples"], 1)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["accuracy"], 1.0)

    def test_transition_constraints_do_not_invent_pass_precision(self):
        gt = {
            "video": {"end_time_sec": 10.0},
            "scoring_blocks": [
                {
                    "id": "P1",
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "gameplay": "gameplay",
                    "team_id": "team_a",
                    "human_constraints": {"passes": {"exact": 2}, "turnovers": {"exact": 1}},
                },
                {"id": "NG", "start_sec": 5.0, "end_sec": 10.0, "gameplay": "non_game", "team_id": None},
            ],
        }
        events = [
            {"event_type": "pass_candidate", "timestamp_start_sec": 1.0, "timestamp_end_sec": 1.1},
            {"event_type": "pass_candidate", "timestamp_start_sec": 2.0, "timestamp_end_sec": 2.1},
            {"event_type": "turnover_candidate", "timestamp_start_sec": 3.0, "timestamp_end_sec": 3.1},
        ]
        result = score_transition_constraints(gt, events)
        self.assertEqual(result["passes"]["positive_block_recall"], 1.0)
        self.assertFalse(result["passes"]["precision_recall_ready"])
        self.assertEqual(result["turnovers"]["positive_block_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
