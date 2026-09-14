from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


def _analyzer() -> TeamOnlyAnalyzer:
    config = {
        "metadata": {
            "home_team": "FC Porto",
            "away_team": "Ovarense",
            "active_kits": {
                "FC Porto": "home",
                "Ovarense": "away",
            },
        },
        "teams": {
            "FC Porto": {
                "kits": {"home": ["Blue", "White", "Red"]},
                "roster": {},
            },
            "Ovarense": {
                "kits": {"away": ["Black", "White"]},
                "roster": {},
            },
        },
    }
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    try:
        json.dump(config, tmp)
        tmp.close()
        return TeamOnlyAnalyzer(tmp.name)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def _solid_hsv_crop(h: int, s: int, v: int) -> np.ndarray:
    hsv = np.zeros((120, 70, 3), dtype=np.uint8)
    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_dark_blue_hue_is_black_only() -> None:
    hsv = np.zeros((10, 10, 3), dtype=np.uint8)
    hsv[:, :, 0] = 110
    hsv[:, :, 1] = 85
    hsv[:, :, 2] = 60
    masks = TeamOnlyAnalyzer._color_masks(hsv)
    assert bool(np.all(masks["Black"]))
    assert not bool(np.any(masks["Blue"]))


def test_black_kit_prefers_ovarenese() -> None:
    result = _analyzer().score_teams(_solid_hsv_crop(110, 80, 55))
    assert result["best_team"] == "Ovarense"
    assert result["evidence_accepted"] is True
    assert result["scores"]


def test_blue_kit_prefers_porto() -> None:
    result = _analyzer().score_teams(_solid_hsv_crop(110, 200, 165))
    assert result["best_team"] == "FC Porto"
    assert result["evidence_accepted"] is True
    assert result["scores"]


def test_ambiguous_crop_does_not_emit_temporal_evidence() -> None:
    result = _analyzer().score_teams(_solid_hsv_crop(0, 0, 125))
    assert result["best_team"] is None
    assert result["evidence_accepted"] is False
    assert result["scores"] == {}
    assert set(result["raw_scores"]) == {"FC Porto", "Ovarense"}


def main() -> None:
    test_dark_blue_hue_is_black_only()
    test_black_kit_prefers_ovarenese()
    test_blue_kit_prefers_porto()
    test_ambiguous_crop_does_not_emit_temporal_evidence()
    print("p1_team_identity_correctness_test: PASS")


if __name__ == "__main__":
    main()
