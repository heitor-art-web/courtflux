from __future__ import annotations

import unittest

import numpy as np

from evaluation.p2_team_colour_crop_audit import _box_for_track, _team_torso_bounds


class TeamColourCropAuditTest(unittest.TestCase):
    def test_box_for_track_uses_matching_prediction(self) -> None:
        row = {
            "team_predictions": [
                {"track_id": 11, "box": [1, 2, 30, 40]},
                {"track_id": 14, "box": [10, 20, 50, 80]},
            ]
        }
        self.assertEqual(_box_for_track(row, 14), [10, 20, 50, 80])

    def test_torso_bounds_match_team_only_analyzer_primary_roi(self) -> None:
        player = np.zeros((100, 200, 3), dtype=np.uint8)
        # TeamOnlyAnalyzer uses int(h * 0.58), and 100 * 0.58 is represented
        # just below 58.0 in binary floating point, so truncation yields 57.
        self.assertEqual(_team_torso_bounds(player), (44, 22, 156, 57))


if __name__ == "__main__":
    unittest.main()
