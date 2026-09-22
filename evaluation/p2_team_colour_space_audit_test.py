from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.p2_team_colour_space_audit import diagnose_torso
from services.vision.team_only_analyzer import TeamOnlyAnalyzer


class TeamColourSpaceAuditTest(unittest.TestCase):
    def _analyzer(self) -> TeamOnlyAnalyzer:
        payload = {
            "metadata": {
                "home_team": "team_a",
                "away_team": "team_b",
                "active_kits": {"team_a": "current", "team_b": "current"},
            },
            "teams": {
                "team_a": {"kits": {"current": ["Green", "White"]}, "roster": {}},
                "team_b": {"kits": {"current": ["Red"]}, "roster": {}},
            },
        }
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        try:
            json.dump(payload, handle)
            handle.close()
            return TeamOnlyAnalyzer(handle.name)
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def test_solid_red_explains_red_prototype_and_team_b(self) -> None:
        analyzer = self._analyzer()
        # OpenCV input is BGR; this is the canonical legacy Red RGB (220, 28, 36).
        torso = np.full((32, 40, 3), (36, 28, 220), dtype=np.uint8)

        result = diagnose_torso(analyzer, torso)

        self.assertEqual(result["total_pixels"], 32 * 40)
        self.assertEqual(
            result["discriminative_pixels"] + result["abstained_pixels"],
            result["total_pixels"],
        )
        self.assertGreater(result["votes_by_team"]["team_b"], 0)
        self.assertEqual(result["votes_by_team"]["team_a"], 0)
        self.assertGreater(result["raw_scores"]["team_b"], result["raw_scores"]["team_a"])
        self.assertEqual(
            result["nearest_unweighted_prototype_all_pixels"],
            {"team_b:Red:0": 32 * 40},
        )
        red = next(row for row in result["prototypes"] if row["key"] == "team_b:Red:0")
        self.assertEqual(red["discriminative_vote_pixels"], 32 * 40)
        self.assertAlmostEqual(red["distance_median"], 0.0, places=3)

    def test_pixel_accounting_distinguishes_low_affinity_and_low_margin(self) -> None:
        analyzer = self._analyzer()
        torso = np.zeros((20, 20, 3), dtype=np.uint8)
        torso[:, :10] = (36, 28, 220)      # red
        torso[:, 10:] = (128, 128, 128)    # grey / likely weak or ambiguous

        result = diagnose_torso(analyzer, torso)

        self.assertEqual(
            result["discriminative_pixels"] + result["abstained_pixels"],
            result["total_pixels"],
        )
        self.assertEqual(
            result["rejected_low_affinity_pixels"] + result["rejected_low_margin_pixels"],
            result["abstained_pixels"],
        )
        self.assertEqual(
            sum(result["votes_by_team"].values()),
            result["discriminative_pixels"],
        )
        self.assertEqual(
            sum(result["votes_by_prototype"].values()),
            result["discriminative_pixels"],
        )


if __name__ == "__main__":
    unittest.main()
