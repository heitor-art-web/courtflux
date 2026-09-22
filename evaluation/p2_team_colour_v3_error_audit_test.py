from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.p2_team_colour_v3_error_audit import diagnose_v3
from services.vision.team_only_analyzer import TeamOnlyAnalyzer


class TeamColourV3ErrorAuditTest(unittest.TestCase):
    def _analyzer(self) -> TeamOnlyAnalyzer:
        payload = {
            "metadata": {
                "home_team": "team_a",
                "away_team": "team_b",
                "active_kits": {"team_a": "current", "team_b": "current"},
            },
            "teams": {
                "team_a": {"kits": {"current": ["White", "Blue"]}, "roster": {}},
                "team_b": {"kits": {"current": ["Green"]}, "roster": {}},
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
    def _solid_rgb(rgb: tuple[int, int, int], height: int = 48, width: int = 64) -> np.ndarray:
        red, green, blue = rgb
        return np.full((height, width, 3), (blue, green, red), dtype=np.uint8)

    def test_audit_reconstructs_current_scorer_raw_scores(self) -> None:
        analyzer = self._analyzer()
        torso = self._solid_rgb((24, 105, 55))

        audit = diagnose_v3(analyzer, torso)
        scored = analyzer._team_scores(
            __import__("cv2").cvtColor(torso, __import__("cv2").COLOR_BGR2LAB)
            .reshape(-1, 3)
            .astype(np.float32)
        )[0]

        self.assertEqual(audit["colour_space"], "CIELAB_continuous_chroma_luminance_v3")
        self.assertAlmostEqual(audit["raw_scores"]["team_a"], scored["team_a"], places=6)
        self.assertAlmostEqual(audit["raw_scores"]["team_b"], scored["team_b"], places=6)
        self.assertEqual(
            audit["discriminative_pixels"] + audit["abstained_pixels"],
            audit["total_pixels"],
        )

    def test_chromatic_and_neutral_prototypes_report_native_components(self) -> None:
        analyzer = self._analyzer()
        audit = diagnose_v3(analyzer, self._solid_rgb((24, 105, 55)))
        by_label = {row["label"]: row for row in audit["prototypes"]}

        self.assertEqual(by_label["White"]["components"]["model"], "neutral")
        self.assertNotIn("hue_delta_deg", by_label["White"]["components"])

        self.assertEqual(by_label["Blue"]["components"]["model"], "chromatic")
        self.assertIn("hue_delta_deg", by_label["Blue"]["components"])
        self.assertIn("chroma_delta", by_label["Blue"]["components"])

        self.assertEqual(by_label["Green"]["components"]["model"], "chromatic")
        self.assertIn("hue_delta_deg", by_label["Green"]["components"])
        self.assertGreater(by_label["Green"]["all_pixel_weighted_winner_count"], 0)


if __name__ == "__main__":
    unittest.main()
