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
    affinity to those prototypes. Arbitrary colours can be supplied as ``#RRGGBB``,
    ``{"rgb": [r, g, b]}`` or ``{"hsv": [degrees, saturation_0_1, value_0_1]}``.

    Legacy colour-name strings remain supported as configuration aliases only. They are
    converted to numeric prototypes before inference; no named hard-mask participates
    in the team decision.

    Chromatic prototypes use hue/chroma-dominant affinity with deliberately weak
    lightness dependence so shadows, compression and exposure changes do not turn one
    kit colour into another. Near-neutral prototypes (white/grey/black) instead require
    low chroma plus compatible lightness. Team scoring remains discriminative per pixel:
    every torso pixel compares all teams on the same evidence and ambiguous pixels can
    abstain.

    Spatially, the classifier prefers a central jersey core inside the existing chest
    ROI. The exact same acceptance thresholds are applied to that core; the full chest
    is used only when the core abstains. This limits arms/background/adjacent-player
    contamination without weakening any colour or Gate-C threshold.
    """

    # These are classifier-internal development parameters, not Gate-C thresholds.
    MIN_TEAM_SCORE = 0.58
    MIN_TEAM_MARGIN = 0.16
    MIN_DISCRIMINATIVE_PIXEL_FRACTION = 0.02
    PIXEL_MIN_AFFINITY = 0.20
    PIXEL_MIN_MARGIN = 0.04

    # LAB_SIGMA is retained as a compatibility/diagnostic reference for older tooling.
    # Team affinity no longer uses isotropic 3-D Lab distance.
    LAB_SIGMA = 34.0
    CHROMATIC_PROTOTYPE_MIN_CHROMA = 18.0
    CHROMATIC_HUE_SIGMA_RAD = 0.62
    CHROMATIC_CHROMA_SIGMA = 70.0
    CHROMATIC_LIGHTNESS_SIGMA = 110.0
    CHROMATIC_PIXEL_RELIABLE_CHROMA = 18.0
    NEUTRAL_CHROMA_SIGMA = 20.0
    NEUTRAL_LIGHTNESS_SIGMA = 55.0

    TOP_PIXEL_FRACTION = 0.35
    SHARED_COLOUR_FLOOR = 0.12
    DISTINCTIVENESS_SIGMA = 38.0
    EMBEDDING_BINS = 4

    # Relative to the existing chest ROI, not the full player crop. These bounds were
    # selected from the prediction-blind spatial audit contract: center 50% of width and
    # middle 60% of height. They are a geometry policy, not colour-specific tuning.
    CORE_X_MIN = 0.25
    CORE_X_MAX = 0.75
    CORE_Y_MIN = 0.20
    CORE_Y_MAX = 0.80
    SPATIAL_POLICY = "central_jersey_core_first_v1"

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

    # Identity-v2 already consumes these keys as a fixed-length vector. They remain a
    # continuous anchor basis: every crop is softly projected onto numeric prototypes.
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
        """Continuous Lab affinity robust to illumination for chromatic colours.

        OpenCV stores Lab channels in 0..255 with neutral a=b=128. For chromatic
        prototypes we compare hue direction strongly, chroma magnitude softly, and
        lightness very softly. Near-neutral prototypes use a separate low-chroma plus
        lightness model so white/grey/black cannot win merely because a coloured jersey
        is dark or desaturated by the camera.
        """
        pixels = lab_pixels.astype(np.float32)
        prototype = prototype_lab.astype(np.float32).reshape(3)

        pixel_l = pixels[:, 0]
        pixel_ab = pixels[:, 1:3] - 128.0
        pixel_chroma = np.linalg.norm(pixel_ab, axis=1)

        prototype_ab = prototype[1:3] - 128.0
        prototype_chroma = float(np.linalg.norm(prototype_ab))
        lightness_delta = np.abs(pixel_l - float(prototype[0]))

        if prototype_chroma < cls.CHROMATIC_PROTOTYPE_MIN_CHROMA:
            chroma_affinity = np.exp(
                -0.5 * np.square(pixel_chroma / max(cls.NEUTRAL_CHROMA_SIGMA, 1e-6))
            )
            lightness_affinity = np.exp(
                -0.5
                * np.square(
                    lightness_delta / max(cls.NEUTRAL_LIGHTNESS_SIGMA, 1e-6)
                )
            )
            return chroma_affinity * lightness_affinity

        denominator = np.maximum(pixel_chroma * prototype_chroma, 1e-6)
        cosine = np.sum(pixel_ab * prototype_ab.reshape(1, 2), axis=1) / denominator
        cosine = np.clip(cosine, -1.0, 1.0)
        hue_delta = np.arccos(cosine)
        hue_affinity = np.exp(
            -0.5
            * np.square(
                hue_delta / max(cls.CHROMATIC_HUE_SIGMA_RAD, 1e-6)
            )
        )
        chroma_delta = np.abs(pixel_chroma - prototype_chroma)
        chroma_affinity = np.exp(
            -0.5
            * np.square(
                chroma_delta / max(cls.CHROMATIC_CHROMA_SIGMA, 1e-6)
            )
        )
        lightness_affinity = np.exp(
            -0.5
            * np.square(
                lightness_delta / max(cls.CHROMATIC_LIGHTNESS_SIGMA, 1e-6)
            )
        )
        chroma_reliability = np.clip(
            pixel_chroma / max(cls.CHROMATIC_PIXEL_RELIABLE_CHROMA, 1e-6),
            0.0,
            1.0,
        )
        return hue_affinity * chroma_affinity * lightness_affinity * chroma_reliability

    def _team_scores(
        self,
        lab_pixels: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float], float, int]:
        """Score teams from one shared discriminative pixel pool."""
        team_names = [str(team) for team in self.active_colour_prototypes]
        prototype_affinity: dict[str, float] = {}
        if not team_names or lab_pixels.size == 0:
            return ({name: 0.0 for name in team_names}, prototype_affinity, 0.0, 0)

        team_affinities: list[np.ndarray] = []
        for team_name in team_names:
            prototypes = self.active_colour_prototypes.get(team_name, [])
            if not prototypes:
                team_affinities.append(np.zeros((lab_pixels.shape[0],), dtype=np.float32))
                continue

            weighted_affinities: list[np.ndarray] = []
            weights = self.prototype_weights.get(team_name, [1.0] * len(prototypes))
            for index, prototype in enumerate(prototypes):
                affinity = self._affinity(lab_pixels, prototype["lab"])
                weight = float(weights[index])
                weighted_affinities.append(affinity * weight)
                prototype_affinity[f"{team_name}:{prototype['label']}:{index}"] = self._top_mean(affinity)
            team_affinities.append(np.maximum.reduce(weighted_affinities))

        matrix = np.stack(team_affinities, axis=1)
        best_indices = np.argmax(matrix, axis=1)
        best_values = matrix[np.arange(matrix.shape[0]), best_indices]

        if matrix.shape[1] > 1:
            sorted_values = np.partition(matrix, matrix.shape[1] - 2, axis=1)
            second_values = sorted_values[:, -2]
        else:
            second_values = np.zeros_like(best_values)

        discriminative = (
            (best_values >= self.PIXEL_MIN_AFFINITY)
            & ((best_values - second_values) >= self.PIXEL_MIN_MARGIN)
        )
        discriminative_count = int(np.count_nonzero(discriminative))
        evidence_fraction = discriminative_count / max(int(matrix.shape[0]), 1)

        masses = np.zeros((len(team_names),), dtype=np.float64)
        if discriminative_count:
            for team_index in range(len(team_names)):
                team_mask = discriminative & (best_indices == team_index)
                if np.any(team_mask):
                    masses[team_index] = float(np.sum(best_values[team_mask]))

        total_mass = float(np.sum(masses))
        if total_mass <= 1e-12:
            scores = {team_name: 0.0 for team_name in team_names}
        else:
            scores = {
                team_name: float(masses[index] / total_mass)
                for index, team_name in enumerate(team_names)
            }
        return scores, prototype_affinity, evidence_fraction, discriminative_count

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

    def _accepted_team(
        self,
        raw_scores: dict[str, float],
        evidence_fraction: float,
    ) -> tuple[str | None, float, float]:
        ordered = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
        best_team, best_score = ordered[0] if ordered else (None, 0.0)
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = max(0.0, float(best_score) - float(second_score))
        accepted = (
            str(best_team)
            if best_team is not None
            and evidence_fraction >= self.MIN_DISCRIMINATIVE_PIXEL_FRACTION
            and float(best_score) >= self.MIN_TEAM_SCORE
            and margin >= self.MIN_TEAM_MARGIN
            else None
        )
        return accepted, float(best_score), margin

    def _score_lab_region(self, lab: np.ndarray) -> dict[str, Any] | None:
        pixels = lab.reshape(-1, 3).astype(np.float32)
        if pixels.size == 0:
            return None
        raw_scores, prototype_affinity, evidence_fraction, discriminative_count = self._team_scores(pixels)
        accepted, best_score, margin = self._accepted_team(raw_scores, evidence_fraction)
        return {
            "lab": lab,
            "pixels": pixels,
            "raw_scores": raw_scores,
            "prototype_affinity": prototype_affinity,
            "evidence_fraction": evidence_fraction,
            "discriminative_count": discriminative_count,
            "accepted": accepted,
            "best_score": best_score,
            "margin": margin,
        }

    def score_teams(self, crop: np.ndarray) -> dict:
        """Return selective team evidence using continuous discriminative CIELAB affinity."""
        colour_space = "CIELAB_continuous_chroma_luminance_v3"
        empty = {
            "scores": {},
            "raw_scores": {str(team): 0.0 for team in self.rosters},
            "best_team": None,
            "best_score": 0.0,
            "margin": 0.0,
            "evidence_accepted": False,
            "discriminative_pixel_fraction": 0.0,
            "discriminative_pixels": 0,
            "color_coverage": {},
            "color_embedding": [],
            "prototype_affinity": {},
            "active_kits": self.active_kits,
            "active_colour_prototypes": self._public_prototypes(),
            "colour_space": colour_space,
            "spatial_policy": self.SPATIAL_POLICY,
            "evidence_region": None,
        }
        if crop is None or crop.size == 0 or not self.rosters:
            return empty

        h, w = crop.shape[:2]
        if h < 10 or w < 10:
            return empty

        # Primary chest ROI. Shorts and lettering remain audit context, not implicit votes.
        chest = crop[int(h * 0.22):int(h * 0.58), int(w * 0.22):int(w * 0.78)]
        if chest.size == 0:
            chest = crop[int(h * 0.18):int(h * 0.62), int(w * 0.18):int(w * 0.82)]
        if chest.size == 0:
            chest = crop[: int(h * 0.65), :]
        if chest.size == 0:
            return empty

        full_lab = cv2.cvtColor(chest, cv2.COLOR_BGR2LAB)
        full_result = self._score_lab_region(full_lab)
        if full_result is None:
            return empty

        chest_h, chest_w = chest.shape[:2]
        core_y1 = max(0, min(chest_h - 1, int(round(chest_h * self.CORE_Y_MIN))))
        core_y2 = max(core_y1 + 1, min(chest_h, int(round(chest_h * self.CORE_Y_MAX))))
        core_x1 = max(0, min(chest_w - 1, int(round(chest_w * self.CORE_X_MIN))))
        core_x2 = max(core_x1 + 1, min(chest_w, int(round(chest_w * self.CORE_X_MAX))))
        core = chest[core_y1:core_y2, core_x1:core_x2]
        core_result = (
            self._score_lab_region(cv2.cvtColor(core, cv2.COLOR_BGR2LAB))
            if core.size
            else None
        )

        if core_result is not None and core_result["accepted"] is not None:
            selected = core_result
            evidence_region = "inner_core"
        else:
            selected = full_result
            evidence_region = "full_fallback"

        raw_scores = dict(selected["raw_scores"])
        prototype_affinity = dict(selected["prototype_affinity"])
        evidence_fraction = float(selected["evidence_fraction"])
        discriminative_count = int(selected["discriminative_count"])
        accepted = selected["accepted"]
        best_score = float(selected["best_score"])
        margin = float(selected["margin"])
        selected_lab = selected["lab"]
        selected_pixels = selected["pixels"]

        evidence_scores = raw_scores if accepted is not None else {}
        basis_projection = self._continuous_basis_projection(selected_pixels)

        return {
            "scores": {name: round(value, 6) for name, value in evidence_scores.items()},
            "raw_scores": {name: round(value, 6) for name, value in raw_scores.items()},
            "best_team": accepted,
            "best_score": round(best_score, 6),
            "margin": round(margin, 6),
            "evidence_accepted": accepted is not None,
            "discriminative_pixel_fraction": round(float(evidence_fraction), 6),
            "discriminative_pixels": int(discriminative_count),
            # Backward-compatible field consumed by Identity v2. Values are continuous
            # anchor affinities, not binary/named masks, over the selected evidence region.
            "color_coverage": {name: round(value, 6) for name, value in basis_projection.items()},
            "color_embedding": self._colour_embedding(selected_lab),
            "prototype_affinity": {
                name: round(value, 6) for name, value in prototype_affinity.items()
            },
            "active_kits": self.active_kits,
            "active_colour_prototypes": self._public_prototypes(),
            "colour_space": colour_space,
            "spatial_policy": self.SPATIAL_POLICY,
            "evidence_region": evidence_region,
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
