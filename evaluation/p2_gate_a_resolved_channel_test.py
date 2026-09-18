from __future__ import annotations

import unittest

from evaluation.p2_gate_a_scoreboard import (
    CHALLENGE_CLIPS,
    DEVELOPMENT_CLIPS,
    _challenge_report,
    _predicted_intervals,
    _select_clips,
)


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

    def test_v2_is_challenge_not_blocking_development(self) -> None:
        clips = [
            {"clip": "V1"},
            {"clip": "V2"},
            {"clip": "V3"},
            {"clip": "V4"},
            {"clip": "V5"},
        ]
        development = _select_clips(clips, DEVELOPMENT_CLIPS)
        challenge = _select_clips(clips, CHALLENGE_CLIPS)
        self.assertEqual([row["clip"] for row in development], ["V1", "V3", "V4", "V5"])
        self.assertEqual([row["clip"] for row in challenge], ["V2"])

    def test_empty_challenge_report_is_non_blocking(self) -> None:
        report = _challenge_report([])
        self.assertFalse(report["blocking"])
        self.assertEqual(report["clips"], [])
        self.assertIsNone(report["aggregate"])
        self.assertIsNone(report["diagnostic_gate"])


if __name__ == "__main__":
    unittest.main()
