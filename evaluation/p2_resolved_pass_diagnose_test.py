from __future__ import annotations

import unittest

from evaluation.p2_resolved_pass_diagnose import diagnose, resolve_passes


class P2ResolvedPassDiagnoseTest(unittest.TestCase):
    def test_backfills_flight_when_distinct_receiver_locks_next_frame(self) -> None:
        states = [
            {
                "frame_id": 10,
                "control_state": "unknown",
                "resolved_control_state": "unknown",
                "flight_active": True,
                "flight_frames": 1,
                "flight_origin_track_id": 7,
                "current_track_id": None,
                "reset_count": 0,
            },
            {
                "frame_id": 11,
                "control_state": "unknown",
                "resolved_control_state": "unknown",
                "flight_active": True,
                "flight_frames": 2,
                "flight_origin_track_id": 7,
                "current_track_id": None,
                "reset_count": 0,
            },
            {
                "frame_id": 12,
                "control_state": "possessed",
                "resolved_control_state": "possessed",
                "flight_active": False,
                "flight_frames": 0,
                "flight_origin_track_id": None,
                "current_track_id": 9,
                "reset_count": 0,
            },
        ]
        rows, windows = resolve_passes(states)
        by_frame = {
            int(row["frame_id"]): row["diagnostic_resolved_control_state"]
            for row in rows
        }
        self.assertEqual(by_frame[10], "pass_flight")
        self.assertEqual(by_frame[11], "pass_flight")
        self.assertEqual(by_frame[12], "possessed")
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["origin_track_id"], 7)
        self.assertEqual(windows[0]["receiver_track_id"], 9)

    def test_shot_resolution_has_precedence(self) -> None:
        states = [
            {
                "frame_id": 20,
                "control_state": "unknown",
                "resolved_control_state": "shot_flight",
                "flight_active": True,
                "flight_frames": 1,
                "flight_origin_track_id": 3,
                "current_track_id": None,
                "reset_count": 0,
            },
            {
                "frame_id": 21,
                "control_state": "unknown",
                "resolved_control_state": "unknown",
                "flight_active": True,
                "flight_frames": 2,
                "flight_origin_track_id": 3,
                "current_track_id": None,
                "reset_count": 0,
            },
            {
                "frame_id": 22,
                "control_state": "possessed",
                "resolved_control_state": "possessed",
                "flight_active": False,
                "flight_frames": 0,
                "flight_origin_track_id": None,
                "current_track_id": 5,
                "reset_count": 0,
            },
        ]
        rows, windows = resolve_passes(states)
        by_frame = {
            int(row["frame_id"]): row["diagnostic_resolved_control_state"]
            for row in rows
        }
        self.assertEqual(by_frame[20], "shot_flight")
        self.assertEqual(by_frame[21], "pass_flight")
        self.assertEqual(windows[0]["shot_resolved_frames_preserved"], 1)

    def test_does_not_backfill_same_handler_recovery(self) -> None:
        states = [
            {
                "frame_id": 30,
                "control_state": "unknown",
                "flight_active": True,
                "flight_frames": 1,
                "flight_origin_track_id": 4,
                "current_track_id": None,
                "reset_count": 0,
            },
            {
                "frame_id": 31,
                "control_state": "possessed",
                "flight_active": False,
                "flight_frames": 0,
                "flight_origin_track_id": None,
                "current_track_id": 4,
                "reset_count": 0,
            },
        ]
        rows, windows = resolve_passes(states)
        self.assertEqual(windows, [])
        self.assertEqual(rows[0]["diagnostic_resolved_control_state"], "unknown")

    def test_report_exposes_live_starting_and_pass_resolved_scores(self) -> None:
        states = [
            {
                "frame_id": 1,
                "control_state": "unknown",
                "resolved_control_state": "unknown",
                "flight_active": True,
                "flight_frames": 1,
                "flight_origin_track_id": 1,
                "current_track_id": None,
                "reset_count": 0,
            },
            {
                "frame_id": 2,
                "control_state": "possessed",
                "resolved_control_state": "possessed",
                "flight_active": False,
                "flight_frames": 0,
                "flight_origin_track_id": None,
                "current_track_id": 2,
                "reset_count": 0,
            },
        ]
        gt = {"control_intervals": [{"state": "pass_flight", "frame_start": 1, "frame_end": 1}]}
        report = diagnose(states, gt)
        self.assertEqual(report["live"]["accuracy"], 0.0)
        self.assertEqual(report["starting_resolved"]["accuracy"], 0.0)
        self.assertEqual(report["pass_resolved"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
