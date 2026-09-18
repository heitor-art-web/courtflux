from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from services.p2.airborne_classifier_v1 import AirborneClassifierV1 as _FrameFeatureClassifier


class AirborneClassifierV1:
    """Causal episode-level airborne classifier.

    Training unit is one contiguous flight hypothesis episode. Runtime accumulates only
    current/past frame evidence for the active flight and evaluates the same episode feature
    contract after every update. Stage 1 resolves AIRBORNE vs CONTINUITY. Stage 2 runs only
    when Stage 1 accepts AIRBORNE and resolves PASS vs SHOT. Event generation is outside
    this component.
    """

    VERSION = "courtflux_airborne_classifier_v1_1_episode"
    CLASSES = ("unresolved", "pass_flight", "shot_flight")
    STAGE1_LABELS = ("continuity", "airborne")
    STAGE2_LABELS = ("pass_flight", "shot_flight")
    FRAME_FEATURE_NAMES = _FrameFeatureClassifier.FEATURE_NAMES
    FEATURE_NAMES = (
        "episode_duration_sec",
        "flight_supported_any",
        "flight_supported_fraction",
        "direct_observation_fraction_latest",
        "displacement_max_player_heights",
        "receiver_evidence_any",
        "receiver_evidence_fraction",
        "raw_best_distinct_any",
        "raw_best_distinct_fraction",
        "raw_best_score_mean",
        "raw_best_score_max",
        "candidate_score_mean",
        "candidate_score_max",
        "shot_active_any",
        "shot_active_fraction",
        "preflight_available_fraction",
        "preflight_direct_fraction",
        "near_x_fraction",
        "above_fraction",
        "rim_distance_min_norm",
        "rim_horizontal_min_widths",
        "rim_relative_y_min_heights",
        "rim_relative_y_max_heights",
        "rim_approach_velocity_best_norm_per_sec",
        "rim_rising_velocity_best_heights_per_sec",
        "approach_duration_max_sec",
        "rising_duration_max_sec",
    )

    def __init__(self, model_path: str | Path | None = None) -> None:
        # Explicit nonexistent path prevents the frame-only extractor from reading the
        # P2_AIRBORNE_MODEL environment variable intended for this episode classifier.
        self._frame_extractor = _FrameFeatureClassifier(
            Path("__courtflux_frame_features_only__")
        )
        self.model_path = Path(model_path) if model_path else None
        if self.model_path is None:
            import os

            env_path = os.getenv("P2_AIRBORNE_MODEL", "").strip()
            self.model_path = Path(env_path) if env_path else None
        self.model: dict[str, Any] | None = None
        self._episode_key: tuple[Any, ...] | None = None
        self._episode: dict[str, Any] | None = None
        if self.model_path is not None and self.model_path.exists():
            self.load(self.model_path)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @classmethod
    def new_accumulator(cls) -> dict[str, Any]:
        return {
            "frames": 0,
            "flight_supported": 0,
            "receiver_evidence": 0,
            "raw_best_distinct": 0,
            "shot_active": 0,
            "preflight_available": 0,
            "preflight_direct": 0,
            "near_x": 0,
            "above": 0,
            "raw_best_score_sum": 0.0,
            "raw_best_score_max": 0.0,
            "candidate_score_sum": 0.0,
            "candidate_score_max": 0.0,
            "direct_observation_fraction_latest": 0.0,
            "episode_duration_sec": 0.0,
            "displacement_max_player_heights": 0.0,
            "rim_distance_min_norm": None,
            "rim_horizontal_min_widths": None,
            "rim_relative_y_min_heights": None,
            "rim_relative_y_max_heights": None,
            "rim_approach_velocity_best_norm_per_sec": 0.0,
            "rim_rising_velocity_best_heights_per_sec": 0.0,
            "approach_duration_max_sec": 0.0,
            "rising_duration_max_sec": 0.0,
        }

    @classmethod
    def update_accumulator(
        cls,
        accumulator: dict[str, Any],
        frame_features: dict[str, float],
    ) -> dict[str, float]:
        acc = accumulator
        acc["frames"] = int(acc.get("frames", 0)) + 1
        frames = max(int(acc["frames"]), 1)

        def value(name: str) -> float:
            return cls._float(frame_features.get(name), 0.0)

        acc["flight_supported"] += int(value("flight_supported") >= 0.5)
        acc["receiver_evidence"] += int(value("receiver_evidence") >= 0.5)
        acc["raw_best_distinct"] += int(value("raw_best_distinct_from_origin") >= 0.5)
        acc["shot_active"] += int(value("shot_active") >= 0.5)
        acc["preflight_available"] += int(value("shot_preflight_available") >= 0.5)
        acc["preflight_direct"] += int(value("shot_preflight_direct_observed") >= 0.5)
        acc["near_x"] += int(value("shot_preflight_near_x") >= 0.5)
        acc["above"] += int(value("shot_preflight_above") >= 0.5)

        raw_score = value("raw_best_score")
        candidate_score = value("candidate_score")
        acc["raw_best_score_sum"] += raw_score
        acc["raw_best_score_max"] = max(float(acc["raw_best_score_max"]), raw_score)
        acc["candidate_score_sum"] += candidate_score
        acc["candidate_score_max"] = max(float(acc["candidate_score_max"]), candidate_score)
        acc["direct_observation_fraction_latest"] = value(
            "flight_direct_observation_fraction"
        )
        acc["episode_duration_sec"] = max(
            float(acc["episode_duration_sec"]), value("flight_duration_sec")
        )
        acc["displacement_max_player_heights"] = max(
            float(acc["displacement_max_player_heights"]),
            value("flight_displacement_player_heights"),
        )

        if value("shot_preflight_available") >= 0.5:
            distance = value("shot_preflight_ball_to_rim_distance_norm")
            horizontal = value("shot_preflight_horizontal_distance_rim_widths")
            relative_y = value("shot_preflight_relative_y_rim_heights")
            if acc["rim_distance_min_norm"] is None:
                acc["rim_distance_min_norm"] = distance
                acc["rim_horizontal_min_widths"] = horizontal
                acc["rim_relative_y_min_heights"] = relative_y
                acc["rim_relative_y_max_heights"] = relative_y
            else:
                acc["rim_distance_min_norm"] = min(
                    float(acc["rim_distance_min_norm"]), distance
                )
                acc["rim_horizontal_min_widths"] = min(
                    float(acc["rim_horizontal_min_widths"]), horizontal
                )
                acc["rim_relative_y_min_heights"] = min(
                    float(acc["rim_relative_y_min_heights"]), relative_y
                )
                acc["rim_relative_y_max_heights"] = max(
                    float(acc["rim_relative_y_max_heights"]), relative_y
                )

        # Negative velocity means moving toward the rim / rising in image coordinates.
        acc["rim_approach_velocity_best_norm_per_sec"] = min(
            float(acc["rim_approach_velocity_best_norm_per_sec"]),
            value("shot_preflight_distance_velocity_norm_per_sec"),
        )
        acc["rim_rising_velocity_best_heights_per_sec"] = min(
            float(acc["rim_rising_velocity_best_heights_per_sec"]),
            value("shot_preflight_vertical_velocity_heights_per_sec"),
        )
        acc["approach_duration_max_sec"] = max(
            float(acc["approach_duration_max_sec"]),
            value("shot_preflight_approach_duration_sec"),
        )
        acc["rising_duration_max_sec"] = max(
            float(acc["rising_duration_max_sec"]),
            value("shot_preflight_rising_duration_sec"),
        )

        result = {
            "episode_duration_sec": float(acc["episode_duration_sec"]),
            "flight_supported_any": 1.0 if acc["flight_supported"] else 0.0,
            "flight_supported_fraction": float(acc["flight_supported"]) / frames,
            "direct_observation_fraction_latest": float(
                acc["direct_observation_fraction_latest"]
            ),
            "displacement_max_player_heights": float(
                acc["displacement_max_player_heights"]
            ),
            "receiver_evidence_any": 1.0 if acc["receiver_evidence"] else 0.0,
            "receiver_evidence_fraction": float(acc["receiver_evidence"]) / frames,
            "raw_best_distinct_any": 1.0 if acc["raw_best_distinct"] else 0.0,
            "raw_best_distinct_fraction": float(acc["raw_best_distinct"]) / frames,
            "raw_best_score_mean": float(acc["raw_best_score_sum"]) / frames,
            "raw_best_score_max": float(acc["raw_best_score_max"]),
            "candidate_score_mean": float(acc["candidate_score_sum"]) / frames,
            "candidate_score_max": float(acc["candidate_score_max"]),
            "shot_active_any": 1.0 if acc["shot_active"] else 0.0,
            "shot_active_fraction": float(acc["shot_active"]) / frames,
            "preflight_available_fraction": float(acc["preflight_available"]) / frames,
            "preflight_direct_fraction": float(acc["preflight_direct"]) / frames,
            "near_x_fraction": float(acc["near_x"]) / frames,
            "above_fraction": float(acc["above"]) / frames,
            "rim_distance_min_norm": cls._float(acc["rim_distance_min_norm"], 0.0),
            "rim_horizontal_min_widths": cls._float(
                acc["rim_horizontal_min_widths"], 0.0
            ),
            "rim_relative_y_min_heights": cls._float(
                acc["rim_relative_y_min_heights"], 0.0
            ),
            "rim_relative_y_max_heights": cls._float(
                acc["rim_relative_y_max_heights"], 0.0
            ),
            "rim_approach_velocity_best_norm_per_sec": float(
                acc["rim_approach_velocity_best_norm_per_sec"]
            ),
            "rim_rising_velocity_best_heights_per_sec": float(
                acc["rim_rising_velocity_best_heights_per_sec"]
            ),
            "approach_duration_max_sec": float(acc["approach_duration_max_sec"]),
            "rising_duration_max_sec": float(acc["rising_duration_max_sec"]),
        }
        return {name: float(result[name]) for name in cls.FEATURE_NAMES}

    def reset(self) -> None:
        self._frame_extractor.reset()
        self._episode_key = None
        self._episode = None

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("version") or "") != self.VERSION:
            raise ValueError(f"airborne model version must be {self.VERSION}")
        if tuple(payload.get("classes", [])) != self.CLASSES:
            raise ValueError("airborne output classes do not match runtime contract")
        if tuple(payload.get("features", [])) != self.FEATURE_NAMES:
            raise ValueError("episode feature order does not match runtime contract")
        stages = dict(payload.get("stages", {}) or {})
        self._validate_stage(stages.get("airborne"), self.STAGE1_LABELS, "airborne")
        self._validate_stage(stages.get("type"), self.STAGE2_LABELS, "type")
        self.model = payload
        self.model_path = Path(path)

    def _validate_stage(
        self,
        stage: Any,
        labels: tuple[str, str],
        name: str,
    ) -> None:
        if not isinstance(stage, dict) or tuple(stage.get("labels", [])) != labels:
            raise ValueError(f"invalid {name} stage contract")
        n = len(self.FEATURE_NAMES)
        means = list(stage.get("feature_mean", []))
        scales = list(stage.get("feature_scale", []))
        weights = list(stage.get("weights", []))
        if len(means) != n or len(scales) != n or len(weights) != n:
            raise ValueError(f"invalid {name} stage feature dimensions")
        try:
            float(stage.get("bias", 0.0))
            threshold = float(stage.get("minimum_probability", 0.55))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name} stage scalars") from exc
        if not 0.5 <= threshold <= 1.0:
            raise ValueError(f"{name} minimum_probability must be in [0.5, 1.0]")

    def _episode_identity(self, frame_state: Any, p2: dict[str, Any]) -> tuple[Any, ...]:
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        frame_id = int(getattr(frame_state, "frame_id", 0))
        frame_start = flight.get("frame_start")
        if frame_start is None:
            frames = max(1, int(self._float(flight.get("flight_frames"), 1.0)))
            frame_start = max(0, frame_id - frames + 1)
        return (
            possession.get("reset_count"),
            frame_start,
            flight.get("origin_track_id"),
        )

    def features(self, *, frame_state: Any, p2: dict[str, Any]) -> dict[str, float]:
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        if not bool(flight.get("active", False)):
            self.reset()
            return {name: 0.0 for name in self.FEATURE_NAMES}

        key = self._episode_identity(frame_state, p2)
        if key != self._episode_key:
            self._frame_extractor.reset()
            self._episode_key = key
            self._episode = self.new_accumulator()
        if self._episode is None:
            self._episode = self.new_accumulator()

        frame_features = self._frame_extractor.features(frame_state=frame_state, p2=p2)
        return self.update_accumulator(self._episode, frame_features)

    def _positive_probability(
        self, stage_name: str, features: dict[str, float]
    ) -> float | None:
        if self.model is None:
            return None
        stage = dict(dict(self.model.get("stages", {}) or {}).get(stage_name, {}) or {})
        means = [float(value) for value in stage["feature_mean"]]
        scales = [max(abs(float(value)), 1e-6) for value in stage["feature_scale"]]
        weights = [float(value) for value in stage["weights"]]
        vector = [
            (float(features[name]) - means[index]) / scales[index]
            for index, name in enumerate(self.FEATURE_NAMES)
        ]
        logit = sum(weight * x for weight, x in zip(weights, vector)) + float(stage["bias"])
        if logit >= 0.0:
            exp_neg = math.exp(-logit)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(logit)
        return exp_pos / (1.0 + exp_pos)

    def classify(self, *, frame_state: Any, p2: dict[str, Any]) -> dict[str, Any]:
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        flight_active = bool(flight.get("active", False))
        features = self.features(frame_state=frame_state, p2=p2)

        label = "unresolved"
        confidence = 0.0
        decision_stage = "no_model"
        stage_probabilities: dict[str, Any] | None = None
        output_probabilities: dict[str, float] | None = None

        airborne_probability = self._positive_probability("airborne", features)
        shot_probability = self._positive_probability("type", features)
        if airborne_probability is not None and shot_probability is not None:
            stage1 = dict(self.model["stages"]["airborne"])
            stage2 = dict(self.model["stages"]["type"])
            airborne_minimum = self._float(stage1.get("minimum_probability"), 0.55)
            type_minimum = self._float(stage2.get("minimum_probability"), 0.55)
            pass_probability = 1.0 - shot_probability
            type_confidence = max(pass_probability, shot_probability)
            stage_probabilities = {
                "airborne": {
                    "continuity": round(1.0 - airborne_probability, 6),
                    "airborne": round(airborne_probability, 6),
                    "minimum_probability": round(airborne_minimum, 6),
                },
                "type": {
                    "pass_flight": round(pass_probability, 6),
                    "shot_flight": round(shot_probability, 6),
                    "minimum_probability": round(type_minimum, 6),
                },
            }
            output_probabilities = {
                "unresolved": 1.0 - airborne_probability,
                "pass_flight": airborne_probability * pass_probability,
                "shot_flight": airborne_probability * shot_probability,
            }
            if not flight_active:
                decision_stage = "not_active_flight"
            elif airborne_probability < airborne_minimum:
                confidence = 1.0 - airborne_probability
                decision_stage = "stage1_continuity"
            elif type_confidence < type_minimum:
                confidence = airborne_probability * type_confidence
                decision_stage = "stage2_abstain"
            elif shot_probability >= 0.5:
                label = "shot_flight"
                confidence = airborne_probability * shot_probability
                decision_stage = "stage2_shot"
            else:
                label = "pass_flight"
                confidence = airborne_probability * pass_probability
                decision_stage = "stage2_pass"

        episode_frames = 0 if self._episode is None else int(self._episode.get("frames", 0))
        return {
            "version": self.VERSION,
            "model_loaded": self.model_loaded,
            "model_path": None if self.model_path is None else str(self.model_path),
            "flight_active": flight_active,
            "episode_key": None if self._episode_key is None else list(self._episode_key),
            "episode_frames": episode_frames,
            "label": label,
            "confidence": round(float(confidence), 6),
            "decision_stage": decision_stage,
            "stage_probabilities": stage_probabilities,
            "probabilities": None
            if output_probabilities is None
            else {key: round(float(value), 6) for key, value in output_probabilities.items()},
            "features": {key: round(float(value), 6) for key, value in features.items()},
        }
