from __future__ import annotations

import unittest

from evaluation.p2_gate_c_raw_colour_validation import (
    _best_raw_team,
    _prediction_for_track,
    _sample_raw_result,
)


class GateCRawColourValidationTest(unittest.TestCase):
    def test_prediction_for_track_requires_rescored_evidence(self) -> None:
        row = {
            "team_predictions": [
                {"track_id": 24, "rescored": False, "raw_scores": {"team_a": 0.9, "team_b": 0.1}},
                {"track_id": 24, "rescored": True, "raw_scores": {"team_a": 0.1, "team_b": 0.9}},
            ]
        }
        prediction = _prediction_for_track(row, 24)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction["raw_scores"]["team_b"], 0.9)

    def test_best_raw_team_returns_margin(self) -> None:
        best, score, margin = _best_raw_team(
            {"raw_scores": {"team_a": 0.013, "team_b": 0.987}}
        )
        self.assertEqual(best, "team_b")
        self.assertAlmostEqual(score or 0.0, 0.987)
        self.assertAlmostEqual(margin or 0.0, 0.974)

    def test_verified_holder_track_overrides_runtime_current_track(self) -> None:
        sample = {
            "state": "possessed",
            "frame_start": 245,
            "frame_end": 245,
            "team_id": "team_b",
            "verified_holder_track_id": 24,
            "source_kind": "human_verified_possession_interval",
        }
        states = [
            {
                "frame_id": 245,
                "control_state": "possessed",
                "current_track_id": 20,
                "team_predictions": [
                    {"track_id": 20, "rescored": True, "raw_scores": {"team_a": 0.9, "team_b": 0.1}},
                    {"track_id": 24, "rescored": True, "raw_scores": {"team_a": 0.013, "team_b": 0.987}},
                ],
            }
        ]
        result = _sample_raw_result(sample, states)
        self.assertEqual(result["raw_observations"], 1)
        self.assertEqual(result["raw_correct"], 1)
        self.assertEqual(result["raw_team_votes"], {"team_b": 1})


if __name__ == "__main__":
    unittest.main()
