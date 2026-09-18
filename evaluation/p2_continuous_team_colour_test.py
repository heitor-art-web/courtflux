from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


class ContinuousTeamColourTest(unittest.TestCase):
    def _analyzer(self, team_a_colours, team_b_colours) -> TeamOnlyAnalyzer:
        payload = {
            "metadata": {
                "home_team": "team_a",
                "away_team": "team_b",
                "active_kits": {"team_a": "current", "team_b": "current"},
            },
            "teams": {
                "team_a": {"kits": {"current": team_a_colours}, "roster": {}},
                "team_b": {"kits": {"current": team_b_colours}, "roster": {}},
            },
        }
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        try:
            json.dump(payload, handle)
            handle.close()
            return TeamOnlyAnalyzer(handle.name)
        finally:
            Path(handle.name).unlink(missing_ok=True)

    @staticmethod
    def _solid_rgb(rgb: tuple[int, int, int], size: int = 96) -> np.ndarray:
        red, green, blue = rgb
        return np.full((size, size, 3), (blue, green, red), dtype=np.uint8)

    def test_arbitrary_hex_colours_do_not_require_named_mask(self) -> None:
        analyzer = self._analyzer(
            [{"name": "electric_magenta", "hex": "#D10A8C"}],
            [{"name": "deep_teal", "hex": "#009DA7"}],
        )

        result = analyzer.score_teams(self._solid_rgb((209, 10, 140)))

        self.assertEqual(result["colour_space"], "CIELAB_continuous_v1")
        self.assertEqual(result["best_team"], "team_a")
        self.assertTrue(result["evidence_accepted"])
        self.assertGreater(result["raw_scores"]["team_a"], result["raw_scores"]["team_b"])

    def test_rgb_and_hsv_specs_share_same_open_contract(self) -> None:
        analyzer = self._analyzer(
            [{"name": "custom_orange", "rgb": [240, 125, 25]}],
            [{"name": "custom_cyan", "hsv": [185, 0.85, 0.80]}],
        )

        orange = analyzer.score_teams(self._solid_rgb((240, 125, 25)))
        cyan = analyzer.score_teams(self._solid_rgb((31, 191, 204)))

        self.assertEqual(orange["best_team"], "team_a")
        self.assertEqual(cyan["best_team"], "team_b")

    def test_multicolour_kit_scores_union_of_numeric_prototypes(self) -> None:
        analyzer = self._analyzer(
            [
                {"name": "orange", "rgb": [240, 125, 25]},
                {"name": "blue", "rgb": [0, 92, 230]},
            ],
            [{"name": "yellow", "rgb": [245, 205, 30]}],
        )
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[:, :50] = (25, 125, 240)  # BGR orange
        crop[:, 50:] = (230, 92, 0)    # BGR blue

        result = analyzer.score_teams(crop)

        self.assertEqual(result["best_team"], "team_a")
        self.assertGreater(result["raw_scores"]["team_a"], result["raw_scores"]["team_b"])

    def test_shared_colour_is_continuously_downweighted(self) -> None:
        analyzer = self._analyzer(
            ["White", {"name": "royal_blue", "hex": "#1557D6"}],
            ["White", {"name": "scarlet", "hex": "#D82032"}],
        )

        result = analyzer.score_teams(self._solid_rgb((21, 87, 214)))
        prototypes = result["active_colour_prototypes"]
        shared_white_weight = prototypes["team_a"][0]["weight"]
        unique_blue_weight = prototypes["team_a"][1]["weight"]

        self.assertEqual(result["best_team"], "team_a")
        self.assertLess(shared_white_weight, unique_blue_weight)

    def test_identity_compatibility_vector_is_continuous_not_mask_based(self) -> None:
        analyzer = self._analyzer(
            [{"name": "orange", "hex": "#F07820"}],
            [{"name": "purple", "hex": "#773AAA"}],
        )

        result = analyzer.score_teams(self._solid_rgb((240, 120, 32)))

        self.assertEqual(
            set(result["color_coverage"]),
            {"Black", "Grey", "White", "Blue", "Navy", "Red", "Green", "Yellow"},
        )
        self.assertGreater(result["color_coverage"]["Red"], 0.0)
        self.assertGreater(result["color_coverage"]["Yellow"], 0.0)
        self.assertEqual(len(result["color_embedding"]), 64)
        self.assertAlmostEqual(sum(result["color_embedding"]), 1.0, places=4)

    def test_unknown_legacy_name_fails_fast_with_numeric_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "#RRGGBB|rgb/hsv"):
            self._analyzer(["InfraredDream"], ["White"])


if __name__ == "__main__":
    unittest.main()
