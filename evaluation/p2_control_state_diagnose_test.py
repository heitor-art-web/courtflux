from __future__ import annotations

import unittest

from evaluation.p2_control_state_diagnose import diagnose


class P2ControlStateDiagnoseTest(unittest.TestCase):
    def test_reports_possession_pass_and_shot_failure_signals(self) -> None:
        states = [
            {
                "frame_id": 0,
                "control_state": "unknown",
                "engine_eligible": False,
                "current_track_id": None,
                "lost_frames": 0,
                "flight_active": False,
                "flight_supported": False,
                "shot_active": False,
                "reset_count": 0,
            },
            {
                "frame_id": 1,
                "control_state": "possessed",
                "engine_eligible": True,
                "current_track_id": 7,
                "lost_frames": 0,
                "flight_active": False,
                "flight_supported": False,
                "shot_active": False,
                "reset_count": 0,
            },
            {
                "frame_id": 2,
                "control_state": "unknown",
                "engine_eligible": True,
                "current_track_id": None,
                "lost_frames": 1,
                "flight_active": False,
                "flight_supported": False,
                "shot_active": False,
                "reset_count": 0,
            },
            {
                "frame_id": 3,
                "control_state": "pass_flight",
                "engine_eligible": True,
                "current_track_id": 7,
                "lost_frames": 2,
                "flight_active": True,
                "flight_supported": True,
                "pass_flight_receiver_evidence": True,
                "shot_active": False,
                "reset_count": 0,
            },
            {
                "frame_id": 4,
                "control_state": "possessed",
                "engine_eligible": True,
                "current_track_id": 8,
                "lost_frames": 0,
                "flight_active": False,
                "flight_supported": False,
                "shot_active": False,
                "reset_count": 0,
            },
            {
                "frame_id": 5,
                "control_state": "shot_flight",
                "engine_eligible": True,
                "current_track_id": None,
                "lost_frames": 0,
                "flight_active": False,
                "flight_supported": False,
                "shot_active": True,
                "reset_count": 0,
            },
        ]
        ground_truth = {
            "control_intervals": [
                {"state": "possessed", "frame_start": 0, "frame_end": 1},
                {"state": "pass_flight", "frame_start": 2, "frame_end": 3},
                {"state": "shot_flight", "frame_start": 4, "frame_end": 5},
            ]
        }

        report = diagnose(states, ground_truth)
        self.assertEqual(report["version"], "courtflux_p2_control_diagnose_v1")
        self.assertEqual(report["ground_truth_intervals"], 3)

        possessed = report["intervals"][0]
        self.assertEqual(possessed["matching_rate"], 0.5)
        self.assertEqual(possessed["engine_ineligible_frames"], 1)
        self.assertEqual(
            possessed["diagnostic_signals"]["possessed_truth_without_handler_frames"],
            1,
        )

        pass_flight = report["intervals"][1]
        self.assertEqual(pass_flight["matching_rate"], 0.5)
        self.assertEqual(pass_flight["flight_supported_frames"], 1)
        self.assertEqual(pass_flight["receiver_evidence_frames"], 1)
        self.assertEqual(
            pass_flight["diagnostic_signals"]["pass_truth_without_locked_origin_frames"],
            1,
        )
        self.assertEqual(
            pass_flight["diagnostic_signals"]["pass_truth_with_receiver_evidence_frames"],
            1,
        )
        self.assertEqual(
            pass_flight["diagnostic_signals"]["pass_truth_predicted_unknown_frames"],
            1,
        )

        shot = report["intervals"][2]
        self.assertEqual(shot["matching_rate"], 0.5)
        self.assertEqual(shot["shot_active_frames"], 1)
        self.assertEqual(
            shot["diagnostic_signals"]["shot_truth_not_predicted_shot_flight_frames"],
            1,
        )
        self.assertEqual(
            shot["diagnostic_signals"]["shot_truth_with_handler_still_locked_frames"],
            1,
        )

    def test_candidate_signals_are_optional_and_reported_when_present(self) -> None:
        states = [
            {
                "frame_id": 10,
                "control_state": "unknown",
                "engine_eligible": True,
                "current_track_id": None,
                "lost_frames": 0,
                "flight_active": True,
                "flight_supported": True,
                "flight_origin_track_id": 7,
                "pass_flight_receiver_evidence": True,
                "shot_active": False,
                "reset_count": 0,
                "candidate_track_id": 3,
                "candidate_frames": 1,
                "candidate_score": 0.61,
                "candidate_threshold": 0.58,
                "candidate_required_frames": 3,
            },
            {
                "frame_id": 11,
                "control_state": "unknown",
                "engine_eligible": True,
                "current_track_id": None,
                "lost_frames": 0,
                "flight_active": True,
                "flight_supported": True,
                "flight_origin_track_id": 4,
                "pass_flight_receiver_evidence": False,
                "shot_active": False,
                "reset_count": 0,
                "candidate_track_id": 4,
                "candidate_frames": 1,
                "candidate_score": 0.40,
                "candidate_threshold": 0.58,
                "candidate_required_frames": 3,
            },
            {
                "frame_id": 12,
                "control_state": "unknown",
                "engine_eligible": True,
                "current_track_id": None,
                "lost_frames": 0,
                "flight_active": True,
                "flight_supported": True,
                "flight_origin_track_id": 4,
                "pass_flight_receiver_evidence": True,
                "shot_active": False,
                "reset_count": 0,
                "candidate_track_id": 5,
                "candidate_frames": 3,
                "candidate_score": 0.70,
                "candidate_threshold": 0.58,
                "candidate_required_frames": 3,
            },
        ]
        ground_truth = {
            "control_intervals": [
                {"state": "possessed", "frame_start": 10, "frame_end": 12},
            ]
        }

        interval = diagnose(states, ground_truth)["intervals"][0]
        candidate = interval["candidate_signals"]
        self.assertEqual(candidate["candidate_present_frames"], 3)
        self.assertEqual(candidate["candidate_track_switches"], 2)
        self.assertEqual(candidate["candidate_below_threshold_frames"], 1)
        self.assertEqual(candidate["candidate_above_threshold_frames"], 2)
        self.assertEqual(candidate["candidate_lock_pending_frames"], 1)
        self.assertEqual(candidate["candidate_lock_ready_frames"], 1)
        self.assertEqual(candidate["candidate_distinct_from_flight_origin_frames"], 2)
        self.assertEqual(candidate["candidate_same_as_flight_origin_frames"], 1)
        self.assertEqual(candidate["receiver_evidence_frames"], 2)
        self.assertEqual(candidate["flight_supported_with_distinct_receiver_frames"], 2)
        self.assertEqual(candidate["flight_supported_without_receiver_frames"], 1)
        self.assertEqual(candidate["max_candidate_score"], 0.7)
        self.assertEqual(candidate["mean_candidate_score"], 0.57)

    def test_correlates_optional_frame_input_signals(self) -> None:
        states = [
            {
                "frame_id": 20,
                "control_state": "unknown",
                "engine_eligible": True,
                "current_track_id": None,
                "lost_frames": 1,
                "flight_active": False,
                "flight_supported": False,
                "shot_active": False,
                "reset_count": 0,
            },
            {
                "frame_id": 21,
                "control_state": "possessed",
                "engine_eligible": True,
                "current_track_id": 9,
                "lost_frames": 0,
                "flight_active": False,
                "flight_supported": False,
                "shot_active": False,
                "reset_count": 0,
            },
        ]
        frames = [
            {
                "frame_id": 20,
                "selected_ball": False,
                "selected_ball_conf": None,
                "ball_source": "predicted_hold",
                "players_final": 0,
                "court_state": {"gameplay_status": "gameplay_limited"},
            },
            {
                "frame_id": 21,
                "selected_ball": True,
                "selected_ball_conf": 0.8,
                "ball_source": "detector_associated",
                "players_final": 8,
                "court_state": {"gameplay_status": "gameplay_analysis"},
            },
        ]
        ground_truth = {
            "control_intervals": [
                {"state": "possessed", "frame_start": 20, "frame_end": 21},
            ]
        }

        report = diagnose(states, ground_truth, frames)
        signals = report["intervals"][0]["p1_input_signals"]
        self.assertEqual(report["frame_rows"], 2)
        self.assertEqual(signals["selected_ball_frames"], 1)
        self.assertEqual(signals["predicted_hold_frames"], 1)
        self.assertEqual(signals["player_zero_frames"], 1)
        self.assertEqual(signals["mean_selected_ball_confidence"], 0.8)
        self.assertEqual(
            signals["gameplay_status_counts"],
            {"gameplay_analysis": 1, "gameplay_limited": 1},
        )


if __name__ == "__main__":
    unittest.main()
