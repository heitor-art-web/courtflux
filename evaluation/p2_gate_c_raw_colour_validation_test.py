from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.p2_gate_c_raw_colour_validation import (
    _best_raw_team,
    _prediction_for_track,
    _recompute_prediction,
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

    def test_zero_raw_mass_is_unattributed(self) -> None:
        best, score, margin = _best_raw_team(
            {"raw_scores": {"team_a": 0.0, "team_b": 0.0}}
        )
        self.assertIsNone(best)
        self.assertEqual(score, 0.0)
        self.assertEqual(margin, 0.0)

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
        self.assertEqual(result["score_source"], "archived_run")

    def test_recompute_prediction_uses_archived_box_but_current_analyzer(self) -> None:
        class FakeAnalyzer:
            def score_teams(self, crop):
                self.crop_shape = crop.shape
                return {
                    "raw_scores": {"team_a": 0.2, "team_b": 0.8},
                    "colour_space": "current",
                }

        analyzer = FakeAnalyzer()
        context = {"video_path": Path("fake.mp4"), "analyzer": analyzer}
        row = {"frame_id": 12}
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        player = np.zeros((40, 20, 3), dtype=np.uint8)

        with patch(
            "evaluation.p2_gate_c_raw_colour_validation._box_for_track",
            return_value=[10, 10, 30, 50],
        ), patch(
            "evaluation.p2_gate_c_raw_colour_validation._timestamp",
            return_value=1.5,
        ), patch(
            "evaluation.p2_gate_c_raw_colour_validation._read_frame",
            return_value=frame,
        ), patch(
            "evaluation.p2_gate_c_raw_colour_validation._safe_crop",
            return_value=(player, [10, 10, 30, 50]),
        ):
            result = _recompute_prediction(row, 24, context)

        self.assertEqual(result["raw_scores"]["team_b"], 0.8)
        self.assertEqual(analyzer.crop_shape, (40, 20, 3))


if __name__ == "__main__":
    unittest.main()
