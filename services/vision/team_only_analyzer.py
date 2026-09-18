from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


class TeamOnlyAnalyzer:
    """Continuous colour-space team classifier with explicit abstention.

    Team identity is independent from colour names. Active kits are represented as one
    or more colour prototypes in CIELAB space and player crops are scored by continuous
    distance to those prototypes. Arbitrary colours can be supplied as ``#RRGGBB``,
    ``{"rgb": [r, g, b]}`` or ``{"hsv": [degrees, saturation_0_1, value_0_1]}``.

    Legacy colour-name strings remain supported as configuration aliases only. They are
    converted to numeric prototypes before inference; no named hard-mask participates
    in the team decision.
    """

    MIN_TEAM_SCORE = 0.45
    MIN_TEAM_MARGIN = 0.08
    LAB_SIGMA = 34.0
    TOP_PIXEL_FRACTION = 0.35
    SHARED_COLOUR_FLOOR = 0.12
    DISTINCTIVENESS_SIGMA = 38.0
    EMBEDDING_BINS = 4

    # Compatibility aliases for existing match configs. This table is not the decision
    # space: callers can configure any colour numerically without adding Python code.
    LEGACY_NAMED_RGB: dict[str, tuple[int, int, int]] = {
        "Black": (0, 0, 0),
        "Grey": (128, 128, 128),
        "Gray": (128, 128, 128),
        "White": (255, 255, 255),
        "Blue": (0, 92, 230),
        "Navy": (18, 32, 92),
        "Red": (220, 28, 36),
        "Green": (28, 150, 72),
        "Yellow": (245, 205, 30),
        "Orange": (240, 125, 25),
        "Purple": (120, 55, 170),
        "Pink": (235, 90, 150),
        "Cyan": (30, 190, 205),
        "Teal": (20, 125, 135),
        "Maroon": (120, 30, 45),
        "Burgundy": (105, 28, 52),
        "Lime": (125, 205, 45),
        "Brown": (125, 80, 45),
        "Gold": (220, 175, 35),
        "Silver": (185, 185, 185),
    }

    # Identity-v2 already consumes these keys as a fixed-length vector. They now form a
    # continuous anchor basis: every crop is softly projected onto the anchors in Lab
    # space. An arbitrary colour therefore still has a meaningful temporal embedding.
    CONTINUOUS_BASIS = ("Black", "Grey", "White", "Blue", "Navy", "Red", "Green", "Yellow")

    def __init__(self, profiling_path: str):
        data = json.loads(Path(profiling_path).read_text(encoding="utf-8"))
        self.metadata = data.get("metadata", {}) or {}
        self.rosters = data.get("teams", data.get("rosters", data)) or {}
        self.active_colour_prototypes = self._resolve_active_colour_prototypes()
        self.active_kits = {
            team: [str(item["label"]) for item in prototypes]
            for team, prototypes in self.active_colour_prototypes.items()
        }
        self.prototype_weights = self._build_prototype_weights()
        self._basis_prototypes = {
            name: self._prototype_from_rgb(name, self.LEGACY_NAMED_RGB[name], source="continuous_basis")
            for name in self.CONTINUOUS_BASIS
        }

    @staticmethod
    def _clean_color(value: object) -> str:
        return str(value).strip().title()

    @staticmethod
    def _parse_hex(value: str) -> tuple[int, int, int]:
        text = value.strip().lstrip("#")
        if len(text) == 3:
            text = "".join(char * 2 for char in text)
        if len(text) != 6:
            raise ValueError(f"invalid hex colour {value!r}; expected #RRGGBB")
        try:
            return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError(f"invalid hex colour {value!r}") from exc

    @staticmethod
    def _validate_rgb(values: Any) -> tuple[int, int, int]:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("rgb colour must be [r, g, b]")
        result = tuple(int(round(float(value))) for value in values)
        if any(value < 0 or value > 255 for value in result):
            raise ValueError("rgb colour channels must be between 0 and 255")
        return result  # type: ignore[return-value]

    @staticmethod
    def _rgb_from_hsv(values: Any) -> tuple[int, int, int]:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError("hsv colour must be [degrees, saturation_0_1, value_0_1]")
        hue, saturation, value = (float(item) for item in values)
        if not 0.0 <= hue <= 360.0 or not 0.0 <= saturation <= 1.0 or not 0.0 <= value <= 1.0:
            raise ValueError("hsv expects H=0..360 and S/V=0..1")
        red, green, blue = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, saturation, value)
        return tuple(int(round(channel * 255.0)) for channel in (red, green, blue))  # type: ignore[return-value]

    @staticmethod
    def _rgb_to_lab(rgb: tuple[int, int, int]) -> np.ndarray:
        red, green, blue = rgb
        bgr = np.array([[[blue, green, red]]], dtype=np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

    def _prototype_from_rgb(
        self,
        label: str,
        rgb: tuple[int, int, int],
        *,
        source: str,
    ) -> dict[str, Any]:
        return {
            "label": label,
            "rgb": [int(value) for value in rgb],
            "lab": self._rgb_to_lab(rgb),
            "source": source,
        }

    def _normalise_colour_spec(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            label = str(value.get("name") or value.get("label") or "custom").strip() or "custom"
            if "rgb" in value:
                return self._prototype_from_rgb(label, self._validate_rgb(value["rgb"]), source="rgb")
            if "hex" in value:
                return self._prototype_from_rgb(label, self._parse_hex(str(value["hex"])), source="hex")
            if "hsv" in value:
                return self._prototype_from_rgb(label, self._rgb_from_hsv(value["hsv"]), source="hsv")
            canonical = self._clean_color(label)
            if canonical in self.LEGACY_NAMED_RGB:
                return self._prototype_from_rgb(canonical, self.LEGACY_NAMED_RGB[canonical], source="legacy_name")
            raise ValueError(
                f"colour spec {value!r} has no numeric prototype; use rgb, hex or hsv for arbitrary colours"
            )

        text = str(value).strip()
        if text.startswith("#"):
            return self._prototype_from_rgb(text, self._parse_hex(text), source="hex")
        canonical = self._clean_color(text)
        if canonical in self.LEGACY_NAMED_RGB:
            if canonical == "Gray":
                canonical = "Grey"
            return self._prototype_from_rgb(canonical, self.LEGACY_NAMED_RGB[canonical], source="legacy_name")
        raise ValueError(
            f"unsupported legacy colour name {text!r}; use #RRGGBB or an rgb/hsv colour spec"
        )

    def _resolve_active_colour_prototypes(self) -> dict[str, list[dict[str, Any]]]:
        explicit = self.metadata.get("active_kits", {}) or {}
        home_team = self.metadata.get("home_team")
        away_team = self.metadata.get("away_team")
        resolved: dict[str, list[dict[str, Any]]] = {}

        for team_name, team_data in self.rosters.items():
            kits = (team_data or {}).get("kits", {}) or {}
            requested = explicit.get(team_name)
            colours: list[Any] = []

            if requested and requested in kits:
                colours = list(kits.get(requested, []) or [])
            elif team_name == home_team and "home" in kits:
                colours = list(kits.get("home", []) or [])
            elif team_name == away_team and "away" in kits:
                colours = list(kits.get("away", []) or [])
            elif kits:
                colours = list(next(iter(kits.values())) or [])
            else:
                colours = list((team_data or {}).get("colors", []) or [])

            resolved[str(team_name)] = [self._normalise_colour_spec(colour) for colour in colours]
        return resolved

    @staticmethod
    def _lab_distance(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)))

    def _build_prototype_weights(self) -> dict[str, list[float]]:
        weights: dict[str, list[float]] = {}
        for team_name, prototypes in self.active_colour_prototypes.items():
            others = [
                other
                for other_team, other_prototypes in self.active_colour_prototypes.items()
                if other_team != team_name
                for other in other_prototypes
            ]
            team_weights: list[float] = []
            for prototype in prototypes:
                if not others:
                    team_weights.append(1.0)
                    continue
                nearest = min(
                    self._lab_distance(prototype["lab"], other["lab"])
                    for other in others
                )
                distinctiveness = 1.0 - math.exp(
                    -0.5 * (nearest / max(self.DISTINCTIVENESS_SIGMA, 1e-6)) ** 2
                )
                team_weights.append(
                    self.SHARED_COLOUR_FLOOR
                    + (1.0 - self.SHARED_COLOUR_FLOOR) * distinctiveness
                )
            weights[team_name] = team_weights
        return weights

    @classmethod
    def _top_mean(cls, values: np.ndarray) -> float:
        flat = values.reshape(-1)
        if flat.size == 0:
            return 0.0
        count = max(1, int(math.ceil(flat.size * cls.TOP_PIXEL_FRACTION)))
        if count >= flat.size:
            return float(np.mean(flat))
        partitioned = np.partition(flat, flat.size - count)
        return float(np.mean(partitioned[-count:]))

    @classmethod
    def _affinity(cls, lab_pixels: np.ndarray, prototype_lab: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(lab_pixels - prototype_lab.reshape(1, 3), axis=1)
        return np.exp(-0.5 * np.square(distance / max(cls.LAB_SIGMA, 1e-6)))

    def _team_scores(
        self,
        lab_pixels: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        scores: dict[str, float] = {}
        prototype_affinity: dict[str, float] = {}

        for team_name, prototypes in self.active_colour_prototypes.items():
            if not prototypes:
                scores[team_name] = 0.0
                continue
            weighted_affinities = []
            for index, prototype in enumerate(prototypes):
                affinity = self._affinity(lab_pixels, prototype["lab"])
                weight = self.prototype_weights.get(team_name, [1.0] * len(prototypes))[index]
                weighted_affinities.append(affinity * weight)
                prototype_affinity[f"{team_name}:{prototype['label']}:{index}"] = self._top_mean(affinity)
            combined = np.maximum.reduce(weighted_affinities)
            scores[team_name] = self._top_mean(combined)
        return scores, prototype_affinity

    def _continuous_basis_projection(self, lab_pixels: np.ndarray) -> dict[str, float]:
        return {
            name: self._top_mean(self._affinity(lab_pixels, prototype["lab"]))
            for name, prototype in self._basis_prototypes.items()
        }

    @classmethod
    def _colour_embedding(cls, lab: np.ndarray) -> list[float]:
        pixels = lab.reshape(-1, 3).astype(np.float32)
        if pixels.size == 0:
            return [0.0] * (cls.EMBEDDING_BINS ** 3)
        hist, _ = np.histogramdd(
            pixels,
            bins=(cls.EMBEDDING_BINS,) * 3,
            range=((0.0, 256.0), (0.0, 256.0), (0.0, 256.0)),
        )
        flat = hist.reshape(-1).astype(np.float64)
        total = float(np.sum(flat))
        if total > 0:
            flat /= total
        return [round(float(value), 6) for value in flat]

    def _public_prototypes(self) -> dict[str, list[dict[str, Any]]]:
        return {
            team: [
                {
                    "label": str(item["label"]),
                    "rgb": list(item["rgb"]),
                    "source": str(item["source"]),
                    "weight": round(float(self.prototype_weights.get(team, [1.0] * len(items))[index]), 6),
                }
                for index, item in enumerate(items)
            ]
            for team, items in self.active_colour_prototypes.items()
        }

    def score_teams(self, crop: np.ndarray) -> dict:
        """Return selective team evidence using continuous CIELAB prototype affinity."""
        empty = {
            "scores": {},
            "raw_scores": {str(team): 0.0 for team in self.rosters},
            "best_team": None,
            "best_score": 0.0,
            "margin": 0.0,
            "evidence_accepted": False,
            "color_coverage": {},
            "color_embedding": [],
            "prototype_affinity": {},
            "active_kits": self.active_kits,
            "active_colour_prototypes": self._public_prototypes(),
            "colour_space": "CIELAB_continuous_v1",
        }
        if crop is None or crop.size == 0 or not self.rosters:
            return empty

        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return empty

        # Bias toward the torso rather than head/skin or shorts. Shorts and lettering are
        # human-audit context, not implicit torso-colour votes.
        chest = crop[int(h * 0.18):int(h * 0.62), int(w * 0.18):int(w * 0.82)]
        if chest.size == 0:
            chest = crop[: int(h * 0.65), :]

        lab = cv2.cvtColor(chest, cv2.COLOR_BGR2LAB)
        lab_pixels = lab.reshape(-1, 3).astype(np.float32)
        if lab_pixels.size == 0:
            return empty

        raw_scores, prototype_affinity = self._team_scores(lab_pixels)
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
        basis_projection = self._continuous_basis_projection(lab_pixels)

        return {
            "scores": {name: round(value, 6) for name, value in evidence_scores.items()},
            "raw_scores": {name: round(value, 6) for name, value in raw_scores.items()},
            "best_team": accepted,
            "best_score": round(best_score, 6),
            "margin": round(margin, 6),
            "evidence_accepted": accepted is not None,
            # Backward-compatible field consumed by Identity v2. Values are continuous
            # anchor affinities, not binary/named masks.
            "color_coverage": {name: round(value, 6) for name, value in basis_projection.items()},
            "color_embedding": self._colour_embedding(lab),
            "prototype_affinity": {
                name: round(value, 6) for name, value in prototype_affinity.items()
            },
            "active_kits": self.active_kits,
            "active_colour_prototypes": self._public_prototypes(),
            "colour_space": "CIELAB_continuous_v1",
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
