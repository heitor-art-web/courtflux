from __future__ import annotations

import unittest

from evaluation.p2_ground_truth_dual import evaluate_payload_dual


class P2GroundTruthDualTest(unittest.TestCase):
    def test_reports_live_and_resolved_without_replacing_live(self) -> None:
        states = [
            {
                "frame_id": 10,
                "control_state": "unknown",
                "live_control_state": "unknown",
                "resolved_control_state": "shot_flight",
                "resolved_control_provisional": True,
                "resolved_shot_backfill_applied": False,
            },
            {
                "frame_id": 11,
                "control_state": "shot_flight",
                "live_control_state": "shot_flight",
                "resolved_control_state": "shot_flight",
                "resolved_control_provisional": True,
                "resolved_shot_backfill_applied": True,
            },
        ]
        ground_truth = {
            "video": {"fps": 25.0},
            "events": [],
            "control_intervals": [
                {"state": "shot_flight", "frame_start": 10, "frame_end": 11}
            ],
        }

        report = evaluate_payload_dual(
            predicted_events=[],
            predicted_states=states,
            ground_truth=ground_truth,
        )

        self.assertEqual(report["control_states"]["accuracy"], 0.5)
        self.assertEqual(report["resolved_control_states"]["accuracy"], 1.0)
        contract = report["control_state_contract"]
        self.assertEqual(contract["resolved_field_rows"], 2)
        self.assertEqual(contract["resolved_field_coverage"], 1.0)
        self.assertEqual(contract["resolved_provisional_frames"], 2)
        self.assertEqual(contract["resolved_shot_backfill_activations"], 1)

    def test_legacy_states_fall_back_to_live_state(self) -> None:
        states = [
            {"frame_id": 20, "control_state": "possessed"},
            {"frame_id": 21, "control_state": "unknown"},
        ]
        ground_truth = {
            "video": {"fps": 25.0},
            "events": [],
            "control_intervals": [
                {"state": "possessed", "frame_start": 20, "frame_end": 21}
            ],
        }

        report = evaluate_payload_dual(
            predicted_events=[],
            predicted_states=states,
            ground_truth=ground_truth,
        )

        self.assertEqual(report["control_states"]["accuracy"], 0.5)
        self.assertEqual(report["resolved_control_states"]["accuracy"], 0.5)
        self.assertEqual(report["control_state_contract"]["resolved_field_rows"], 0)


if __name__ == "__main__":
    unittest.main()
