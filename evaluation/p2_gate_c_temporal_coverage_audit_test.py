from __future__ import annotations

import unittest

from evaluation.p2_gate_c_temporal_coverage_audit import _frame_diagnostic


class GateCTemporalCoverageAuditTest(unittest.TestCase):
    def test_frame_diagnostic_separates_raw_fused_and_profile_votes(self) -> None:
        states = [
            {
                "frame_id": 10,
                "current_track_id": 7,
                "current_team_id": "team_a",
                "team_predictions": [
                    {
                        "track_id": 7,
                        "team": "team_a",
                        "rescored": True,
                        "source": "identity_v2_decay_window",
                        "raw_scores": {"team_a": 0.1, "team_b": 0.9},
                        "fused_best_team": "team_b",
                    }
                ],
            },
            {
                "frame_id": 11,
                "current_track_id": 7,
                "current_team_id": "team_a",
                "team_predictions": [
                    {
                        "track_id": 7,
                        "team": "team_a",
                        "rescored": False,
                        "source": "identity_v2_team_profile",
                    }
                ],
            },
        ]
        result = _frame_diagnostic(states, start=10, end=11, truth_team="team_b")
        self.assertEqual(result["current_team_votes"], {"team_a": 2})
        self.assertEqual(result["rescored"], 1)
        self.assertEqual(result["reused"], 1)
        self.assertEqual(result["raw_best_votes"], {"team_b": 1})
        self.assertEqual(result["fused_best_votes"], {"team_b": 1})
        self.assertEqual(result["prediction_team_votes"], {"team_a": 2})
        self.assertEqual(result["raw_truth_hits"], 1)
        self.assertEqual(result["fused_truth_hits"], 1)


if __name__ == "__main__":
    unittest.main()
