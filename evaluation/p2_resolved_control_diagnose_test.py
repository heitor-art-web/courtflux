from __future__ import annotations

import unittest

from evaluation.p2_resolved_control_diagnose import diagnose, resolve_states


class P2ResolvedControlDiagnoseTest(unittest.TestCase):
    def test_backfills_open_flight_when_shot_activates(self) -> None:
        states = [
            {"frame_id": 10, "control_state": "possessed", "flight_active": False, "flight_frames": 0, "shot_active": False, "reset_count": 0},
            {"frame_id": 11, "control_state": "unknown", "flight_active": True, "flight_frames": 1, "shot_active": False, "reset_count": 0},
            {"frame_id": 12, "control_state": "unknown", "flight_active": True, "flight_frames": 2, "shot_active": False, "reset_count": 0},
            {"frame_id": 13, "control_state": "shot_flight", "flight_active": False, "flight_frames": 0, "shot_active": True, "reset_count": 0},
        ]
        rows, windows = resolve_states(states)
        resolved = {int(row["frame_id"]): row["resolved_control_state"] for row in rows}
        self.assertEqual(resolved[10], "possessed")
        self.assertEqual(resolved[11], "shot_flight")
        self.assertEqual(resolved[12], "shot_flight")
        self.assertEqual(resolved[13], "shot_flight")
        self.assertEqual(windows[0]["flight_start_frame"], 11)
        self.assertEqual(windows[0]["relabeled_frames"], 3)

    def test_receiver_evidence_prevents_backfill(self) -> None:
        states = [
            {"frame_id": 20, "control_state": "unknown", "flight_active": True, "flight_frames": 1, "shot_active": False, "reset_count": 0, "pass_flight_receiver_evidence": True},
            {"frame_id": 21, "control_state": "shot_flight", "flight_active": False, "flight_frames": 0, "shot_active": True, "reset_count": 0},
        ]
        rows, windows = resolve_states(states)
        resolved = {int(row["frame_id"]): row["resolved_control_state"] for row in rows}
        self.assertEqual(resolved[20], "unknown")
        self.assertTrue(windows[0]["receiver_evidence"])
        self.assertEqual(windows[0]["relabeled_frames"], 0)

    def test_report_compares_live_and_resolved(self) -> None:
        states = [
            {"frame_id": 1, "control_state": "unknown", "flight_active": True, "flight_frames": 1, "shot_active": False, "reset_count": 0},
            {"frame_id": 2, "control_state": "shot_flight", "flight_active": False, "flight_frames": 0, "shot_active": True, "reset_count": 0},
        ]
        gt = {"control_intervals": [{"state": "shot_flight", "frame_start": 1, "frame_end": 2}]}
        report = diagnose(states, gt)
        self.assertEqual(report["live"]["accuracy"], 0.5)
        self.assertEqual(report["resolved"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
