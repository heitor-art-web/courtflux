from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from evaluation.p2_team_colour_spatial_roi_audit import (
    _grid_masks,
    _score_lab_pixels,
    analyze_torso_spatial,
)
from services.vision.team_only_analyzer import TeamOnlyAnalyzer


class TeamColourSpatialRoiAuditTest(unittest.TestCase):
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

    def test_grid_masks_partition_every_pixel_exactly_once(self) -> None:
        height, width = 61, 47
        masks = _grid_masks(height, width)

        coverage = np.zeros((height, width), dtype=np.int16)
        for _, mask, _ in masks:
            coverage += mask.astype(np.int16)

        self.assertEqual(len(masks), 9)
        self.assertTrue(np.all(coverage == 1))
        self.assertEqual(int(np.sum(coverage)), height * width)

    def test_spatial_score_matches_current_affinity_mass_scorer(self) -> None:
        analyzer = self._analyzer()
        torso = np.zeros((60, 80, 3), dtype=np.uint8)
        torso[:, :50] = (72, 150, 28)   # BGR green
        torso[:, 50:] = (255, 255, 255) # white
        lab_pixels = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

        audit = _score_lab_pixels(analyzer, lab_pixels)
        production_scores = analyzer._team_scores(lab_pixels)[0]

        self.assertAlmostEqual(audit["raw_scores"]["team_a"], production_scores["team_a"], places=6)
        self.assertAlmostEqual(audit["raw_scores"]["team_b"], production_scores["team_b"], places=6)
        self.assertEqual(
            audit["discriminative_pixels"],
            sum(audit["votes_by_team"].values()),
        )

    def test_inner_core_and_outer_ring_separate_spatial_evidence(self) -> None:
        analyzer = self._analyzer()
        height = width = 90
        torso = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)

        # Match analyze_torso_spatial() inner-core geometry exactly.
        x1, x2 = int(round(width * 0.25)), int(round(width * 0.75))
        y1, y2 = int(round(height * 0.20)), int(round(height * 0.80))
        torso[y1:y2, x1:x2] = (72, 150, 28)  # BGR for canonical Green RGB(28,150,72)

        spatial = analyze_torso_spatial(analyzer, torso)

        full_pixels = spatial["full"]["pixels"]
        inner_pixels = spatial["inner_core"]["score"]["pixels"]
        outer_pixels = spatial["outer_ring"]["pixels"]
        grid_pixels = sum(cell["score"]["pixels"] for cell in spatial["grid_3x3"])

        self.assertEqual(full_pixels, height * width)
        self.assertEqual(inner_pixels + outer_pixels, full_pixels)
        self.assertEqual(grid_pixels, full_pixels)
        self.assertEqual(spatial["inner_core"]["score"]["best_team_raw"], "team_b")
        self.assertEqual(spatial["outer_ring"]["best_team_raw"], "team_a")


if __name__ == "__main__":
    unittest.main()
