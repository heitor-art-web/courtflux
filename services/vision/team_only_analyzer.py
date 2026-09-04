from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class TeamOnlyAnalyzer:
    """Lightweight HSV team classifier used when jersey inference is disabled."""

    def __init__(self, profiling_path: str):
        data = json.loads(Path(profiling_path).read_text(encoding="utf-8"))
        self.rosters = data.get("teams", data.get("rosters", data))

    def get_team_color(self, crop: np.ndarray) -> Optional[str]:
        if crop is None or crop.size == 0 or not self.rosters:
            return None
        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return None

        chest = crop[int(h * 0.12):int(h * 0.50), int(w * 0.20):int(w * 0.80)]
        if chest.size == 0:
            chest = crop[: int(h * 0.6), :]
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        total = chest.shape[0] * chest.shape[1]
        if not total:
            return None

        masks = {
            "Black": (hsv[:, :, 2] < 75) & (hsv[:, :, 1] < 90),
            "Grey": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 165),
            "Gray": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 165),
            "White": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 165),
            "Blue": (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] >= 50),
            "Navy": (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] < 100),
            "Red": ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] > 60),
            "Green": (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50),
            "Yellow": (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 60),
        }

        best_team = None
        best_score = -1.0
        for team_name, team_data in self.rosters.items():
            kits = team_data.get("kits", {}) or {"default": team_data.get("colors", [])}
            team_score = 0.0
            for kit_colors in kits.values():
                pixels = 0
                for color in kit_colors:
                    mask = masks.get(str(color).strip().title())
                    if mask is not None:
                        pixels += int(np.sum(mask))
                team_score = max(team_score, pixels / total)
            if team_score > best_score:
                best_score = team_score
                best_team = team_name
        return best_team if best_score >= 0.10 else None

    def get_shirt_number(self, *args, **kwargs) -> str:
        return ""
