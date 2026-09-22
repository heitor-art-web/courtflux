from __future__ import annotations

import unittest

from evaluation.p2_possession_holder_audit import _compact_row, _track_prediction


class PossessionHolderAuditTest(unittest.TestCase):
    def test_track_prediction_selects_matching_track(self) -> None:
        row = {
            "team_predictions": [
                {"track_id": 7, "raw_scores": {"team_a": 0.9, "team_b": 0.1}},
                {"track_id": 11, "raw_scores": {"team_a": 0.2, "team_b": 0.8}},
            ]
        }
        self.assertEqual(_track_prediction(row, 11)["track_id"], 11)

    def test_compact_row_uses_current_track_team_evidence(self) -> None:
        row = {
            "frame_id": 245,
            "timestamp_sec": 4.9,
            "control_state": "possessed",
            "current_track_id": 11,
            "current_confidence": 0.72,
            "candidate_track_id": 12,
            "candidate_frames": 2,
            "candidate_score": 0.44,
            "raw_best_track_id": 12,
            "raw_best_score": 0.51,
            "raw_best_above_threshold": True,
            "flight_active": False,
            "shot_active": False,
            "team_predictions": [
                {"track_id": 11, "raw_scores": {"team_a": 0.25, "team_b": 0.75}},
                {"track_id": 12, "raw_scores": {"team_a": 0.8, "team_b": 0.2}},
            ],
        }
        compact = _compact_row(row)
        self.assertEqual(compact["current"], 11)
        self.assertEqual(compact["raw_best"], 12)
        self.assertEqual(compact["current_team_raw"], {"team_a": 0.25, "team_b": 0.75})


if __name__ == "__main__":
    unittest.main()
