from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class TeamOnlyAnalyzer:
    """Lightweight HSV team classifier with explicit, comparable team scores.

    The previous beta compared every kit for every team and selected the maximum. That
    made overlapping kits (for example Porto home and Ovarense home sharing white/red)
    systematically favour the first team in the config. This classifier instead uses
    the active match kit, discounts colours shared by both active kits, and exposes the
    raw team scores so the tracker can aggregate evidence across frames.
    """

    def __init__(self, profiling_path: str):
        data = json.loads(Path(profiling_path).read_text(encoding="utf-8"))
        self.metadata = data.get("metadata", {}) or {}
        self.rosters = data.get("teams", data.get("rosters", data)) or {}
        self.active_kits = self._resolve_active_kits()
        self.color_weights = self._build_color_weights()

    @staticmethod
    def _clean_color(value: object) -> str:
        return str(value).strip().title()

    def _resolve_active_kits(self) -> dict[str, list[str]]:
        explicit = self.metadata.get("active_kits", {}) or {}
        home_team = self.metadata.get("home_team")
        away_team = self.metadata.get("away_team")
        resolved: dict[str, list[str]] = {}

        for team_name, team_data in self.rosters.items():
            kits = (team_data or {}).get("kits", {}) or {}
            requested = explicit.get(team_name)
            colors: list[str] = []

            if requested and requested in kits:
                colors = list(kits.get(requested, []) or [])
            elif team_name == home_team and "home" in kits:
                colors = list(kits.get("home", []) or [])
            elif team_name == away_team and "away" in kits:
                colors = list(kits.get("away", []) or [])
            elif kits:
                colors = list(next(iter(kits.values())) or [])
            else:
                colors = list((team_data or {}).get("colors", []) or [])

            resolved[str(team_name)] = [self._clean_color(color) for color in colors]
        return resolved

    def _build_color_weights(self) -> dict[str, float]:
        usage: dict[str, int] = {}
        for colors in self.active_kits.values():
            for color in set(colors):
                usage[color] = usage.get(color, 0) + 1

        # Shared colours remain useful for finding a jersey region, but should not be
        # allowed to decide the team by themselves.
        return {color: (1.0 if count == 1 else 0.12) for color, count in usage.items()}

    @staticmethod
    def _color_masks(hsv: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "Black": (hsv[:, :, 2] < 75) & (hsv[:, :, 1] < 100),
            "Grey": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 165),
            "Gray": (hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 75) & (hsv[:, :, 2] <= 165),
            "White": (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 165),
            "Blue": (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] >= 50),
            "Navy": (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] < 105),
            "Red": ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] > 60),
            "Green": (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 50),
            "Yellow": (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 60),
        }

    def score_teams(self, crop: np.ndarray) -> dict:
        """Return normalized evidence scores for each team plus colour diagnostics."""
        empty = {
            "scores": {str(team): 0.0 for team in self.rosters},
            "best_team": None,
            "best_score": 0.0,
            "margin": 0.0,
            "color_coverage": {},
            "active_kits": self.active_kits,
        }
        if crop is None or crop.size == 0 or not self.rosters:
            return empty

        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return empty

        chest = crop[int(h * 0.12):int(h * 0.52), int(w * 0.18):int(w * 0.82)]
        if chest.size == 0:
            chest = crop[: int(h * 0.6), :]
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        total = float(chest.shape[0] * chest.shape[1])
        if total <= 0:
            return empty

        masks = self._color_masks(hsv)
        coverage = {name: float(np.count_nonzero(mask)) / total for name, mask in masks.items()}

        scores: dict[str, float] = {}
        for team_name, colors in self.active_kits.items():
            weighted = 0.0
            weight_total = 0.0
            for color in colors:
                canonical = "Grey" if color == "Gray" else color
                value = coverage.get(canonical, 0.0)
                weight = self.color_weights.get(canonical, 0.12)
                weighted += weight * value
                weight_total += weight
            scores[team_name] = weighted / max(weight_total, 1e-6)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_team, best_score = ordered[0] if ordered else (None, 0.0)
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = max(0.0, best_score - second_score)

        # A team label must contain some actual jersey-colour evidence and beat the
        # alternative by a small but non-zero margin. The tracker aggregates these
        # scores over time, so ambiguous individual frames can safely remain unknown.
        accepted = best_team if best_score >= 0.025 and margin >= 0.008 else None
        return {
            "scores": {name: round(value, 6) for name, value in scores.items()},
            "best_team": accepted,
            "best_score": round(best_score, 6),
            "margin": round(margin, 6),
            "color_coverage": {name: round(value, 6) for name, value in coverage.items()},
            "active_kits": self.active_kits,
        }

    def get_team_color(self, crop: np.ndarray) -> Optional[str]:
        return self.score_teams(crop).get("best_team")

    def resolve_player_name(self, team_name: str | None, jersey_number: str | None) -> str | None:
        if not team_name or jersey_number is None:
            return None
        team_data = self.rosters.get(team_name, {}) or {}
        player = (team_data.get("roster", {}) or {}).get(str(jersey_number))
        if isinstance(player, dict):
            name = str(player.get("name") or "").strip()
            return name or None
        return None

    def get_shirt_number(self, *args, **kwargs) -> str:
        return ""
