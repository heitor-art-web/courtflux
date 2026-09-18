from __future__ import annotations

import unittest

from evaluation.p2_gate_a_scoreboard import _match_intervals, _predicted_intervals


class GateAP3ReadinessTest(unittest.TestCase):
    def test_partial_same_state_overlap_counts_as_detected_episode(self) -> None:
        truth = [
            {"truth_index": 0, "state": "shot_flight", "frame_start": 100, "frame_end": 140}
        ]
        predicted = [
            {"prediction_index": 0, "state": "shot_flight", "frame_start": 120, "frame_end": 130}
        ]
        result = _match_intervals(truth, predicted, tolerance_frames=8)
        self.assertEqual(result["correct_core_intervals"], 1)
        self.assertEqual(result["core_interval_detection_recall"], 1.0)
        self.assertLess(result["boundary_diagnostics"]["mean_iou"], 0.5)

    def test_wrong_airborne_type_is_confusion_not_unresolved(self) -> None:
        truth = [
            {"truth_index": 0, "state": "pass_flight", "frame_start": 20, "frame_end": 30}
        ]
        predicted = [
            {"prediction_index": 0, "state": "shot_flight", "frame_start": 22, "frame_end": 28}
        ]
        result = _match_intervals(truth, predicted, tolerance_frames=8)
        self.assertEqual(result["core_unresolved_intervals"], 0)
        self.assertEqual(result["by_state"]["pass_flight"]["wrong_type"], 1)
        self.assertEqual(result["critical_confusions"]["pass_interval_as_shot"], 1.0)

    def test_one_prediction_cannot_satisfy_two_truth_episodes(self) -> None:
        truth = [
            {"truth_index": 0, "state": "pass_flight", "frame_start": 10, "frame_end": 10},
            {"truth_index": 1, "state": "pass_flight", "frame_start": 15, "frame_end": 15},
        ]
        predicted = [
            {"prediction_index": 0, "state": "pass_flight", "frame_start": 9, "frame_end": 16}
        ]
        result = _match_intervals(truth, predicted, tolerance_frames=8)
        self.assertEqual(result["by_state"]["pass_flight"]["correct"], 1)
        self.assertEqual(result["by_state"]["pass_flight"]["unresolved"], 1)

    def test_sequence_order_is_conditioned_on_both_episodes_detected(self) -> None:
        truth = [
            {"truth_index": 0, "state": "possessed", "frame_start": 0, "frame_end": 10},
            {"truth_index": 1, "state": "pass_flight", "frame_start": 20, "frame_end": 30},
            {"truth_index": 2, "state": "possessed", "frame_start": 40, "frame_end": 50},
        ]
        predicted = [
            {"prediction_index": 0, "state": "possessed", "frame_start": 2, "frame_end": 8},
            {"prediction_index": 1, "state": "pass_flight", "frame_start": 22, "frame_end": 28},
        ]
        result = _match_intervals(truth, predicted, tolerance_frames=8)
        sequence = result["semantic_transitions"]
        self.assertEqual(sequence["gt_total"], 2)
        self.assertEqual(sequence["eligible_detected"], 1)
        self.assertEqual(sequence["correct_order"], 1)
        self.assertEqual(sequence["order_accuracy"], 1.0)
        self.assertEqual(sequence["detected_transition_coverage"], 0.5)

    def test_predicted_state_rows_are_collapsed_into_intervals(self) -> None:
        rows = [
            {"frame_id": 0, "control_state": "possessed"},
            {"frame_id": 1, "control_state": "possessed"},
            {"frame_id": 2, "control_state": "pass_flight"},
            {"frame_id": 3, "control_state": "pass_flight"},
            {"frame_id": 5, "control_state": "pass_flight"},
        ]
        intervals = _predicted_intervals(rows)
        self.assertEqual(len(intervals), 3)
        self.assertEqual((intervals[0]["frame_start"], intervals[0]["frame_end"]), (0, 1))
        self.assertEqual((intervals[1]["frame_start"], intervals[1]["frame_end"]), (2, 3))
        self.assertEqual((intervals[2]["frame_start"], intervals[2]["frame_end"]), (5, 5))


if __name__ == "__main__":
    unittest.main()
