from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class TeamOnlyAnalyzer:
    """Lightweight HSV team classifier with explicit, comparable team scores.

    The classifier uses the active match kit, discounts colours shared by both active
    kits, and only emits temporal evidence when the current crop is itself selective
    enough to accept. Ambiguous crops remain UNKNOWN instead of contaminating a track's
    persistent team profile.
    """

    MIN_TEAM_SCORE = 0.030
    MIN_TEAM_MARGIN = 0.010

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
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # Very dark pixels are black irrespective of hue/saturation. Camera white balance
        # often gives black kits a blue hue; chromatic masks must therefore exclude this
        # region or a black jersey can simultaneously vote Black and Blue.
        black = value < 72
        chromatic = ~black

        grey = chromatic & (saturation < 45) & (value <= 165)
        return {
            "Black": black,
            "Grey": grey,
            "Gray": grey,
            "White": chromatic & (saturation < 50) & (value > 165),
            "Blue": chromatic
            & (hue >= 95)
            & (hue <= 135)
            & (saturation > 55)
            & (value >= 72),
            "Navy": chromatic
            & (hue >= 95)
            & (hue <= 135)
            & (saturation > 40)
            & (value >= 72)
            & (value < 115),
            "Red": chromatic
            & ((hue <= 10) | (hue >= 160))
            & (saturation > 60),
            "Green": chromatic
            & (hue >= 35)
            & (hue <= 85)
            & (saturation > 50),
            "Yellow": chromatic
            & (hue >= 20)
            & (hue <= 35)
            & (saturation > 60),
        }

    def score_teams(self, crop: np.ndarray) -> dict:
        """Return selective team evidence plus raw colour diagnostics.

        ``scores`` is intentionally empty for ambiguous crops so temporal aggregation
        cannot turn repeated low-quality evidence into a confident but wrong team label.
        ``raw_scores`` preserves the per-team measurements for QA.
        """
        empty = {
            "scores": {},
            "raw_scores": {str(team): 0.0 for team in self.rosters},
            "best_team": None,
            "best_score": 0.0,
            "margin": 0.0,
            "evidence_accepted": False,
            "color_coverage": {},
            "active_kits": self.active_kits,
        }
        if crop is None or crop.size == 0 or not self.rosters:
            return empty

        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return empty

        # Bias toward the torso rather than head/skin or shorts. This region is still
        # deliberately broad enough for partial/angled player boxes.
        chest = crop[int(h * 0.18):int(h * 0.62), int(w * 0.18):int(w * 0.82)]
        if chest.size == 0:
            chest = crop[: int(h * 0.65), :]
        hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
        total = float(chest.shape[0] * chest.shape[1])
        if total <= 0:
            return empty

        masks = self._color_masks(hsv)
        coverage = {name: float(np.count_nonzero(mask)) / total for name, mask in masks.items()}

        raw_scores: dict[str, float] = {}
        for team_name, colors in self.active_kits.items():
            weighted = 0.0
            weight_total = 0.0
            for color in colors:
                canonical = "Grey" if color == "Gray" else color
                value = coverage.get(canonical, 0.0)
                weight = self.color_weights.get(canonical, 0.12)
                weighted += weight * value
                weight_total += weight
            raw_scores[team_name] = weighted / max(weight_total, 1e-6)

        ordered = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        best_team, best_score = ordered[0] if ordered else (None, 0.0)
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = max(0.0, best_score - second_score)

        accepted = (
            best_team
            if best_score >= self.MIN_TEAM_SCORE and margin >= self.MIN_TEAM_MARGIN
            else None
        )
        evidence_scores = raw_scores if accepted is not None else {}
        return {
            "scores": {name: round(value, 6) for name, value in evidence_scores.items()},
            "raw_scores": {name: round(value, 6) for name, value in raw_scores.items()},
            "best_team": accepted,
            "best_score": round(best_score, 6),
            "margin": round(margin, 6),
            "evidence_accepted": accepted is not None,
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
