from __future__ import annotations

import unittest

from evaluation.p2_gate_a_scoreboard import _predicted_intervals


class GateAResolvedChannelTest(unittest.TestCase):
    def test_resolved_control_state_is_primary_gate_channel(self) -> None:
        rows = [
            {
                "frame_id": 10,
                "control_state": "unknown",
                "live_control_state": "unknown",
                "resolved_control_state": "shot_flight",
            },
            {
                "frame_id": 11,
                "control_state": "unknown",
                "live_control_state": "unknown",
                "resolved_control_state": "shot_flight",
            },
        ]
        intervals = _predicted_intervals(rows)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["state"], "shot_flight")
        self.assertEqual(intervals[0]["frame_start"], 10)
        self.assertEqual(intervals[0]["frame_end"], 11)

    def test_historical_rows_fall_back_to_live_control_state(self) -> None:
        rows = [
            {"frame_id": 1, "live_control_state": "possessed", "control_state": "unknown"},
            {"frame_id": 2, "live_control_state": "possessed", "control_state": "unknown"},
        ]
        intervals = _predicted_intervals(rows)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["state"], "possessed")

    def test_legacy_rows_fall_back_to_control_state(self) -> None:
        rows = [
            {"frame_id": 1, "control_state": "pass_flight"},
            {"frame_id": 2, "control_state": "pass_flight"},
        ]
        intervals = _predicted_intervals(rows)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["state"], "pass_flight")


if __name__ == "__main__":
    unittest.main()
