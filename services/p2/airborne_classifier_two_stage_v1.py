from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from services.p2.airborne_classifier_v1 import AirborneClassifierV1 as _FeatureAirborneClassifierV1


class AirborneClassifierV1(_FeatureAirborneClassifierV1):
    """Two-stage causal airborne classifier.

    Stage 1 distinguishes AIRBORNE from CONTINUITY using only current/past features.
    Stage 2 runs only after Stage 1 accepts AIRBORNE and distinguishes PASS from SHOT.
    Event generation remains outside this component.
    """

    VERSION = "courtflux_airborne_classifier_v1_two_stage"
    STAGE1_LABELS = ("continuity", "airborne")
    STAGE2_LABELS = ("pass_flight", "shot_flight")

    def load(self, path: str | Path) -> None:
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if str(payload.get("version") or "") != self.VERSION:
            raise ValueError(f"airborne model version must be {self.VERSION}")
        if tuple(payload.get("classes", [])) != self.CLASSES:
            raise ValueError("airborne output classes do not match runtime contract")
        if tuple(payload.get("features", [])) != self.FEATURE_NAMES:
            raise ValueError("airborne model feature order does not match runtime contract")

        stages = dict(payload.get("stages", {}) or {})
        self._validate_stage(stages.get("airborne"), self.STAGE1_LABELS, "airborne")
        self._validate_stage(stages.get("type"), self.STAGE2_LABELS, "type")
        self.model = payload
        self.model_path = Path(path)

    def _validate_stage(
        self,
        stage: Any,
        expected_labels: tuple[str, str],
        stage_name: str,
    ) -> None:
        if not isinstance(stage, dict):
            raise ValueError(f"missing {stage_name} stage")
        if tuple(stage.get("labels", [])) != expected_labels:
            raise ValueError(f"{stage_name} stage labels do not match runtime contract")
        n_features = len(self.FEATURE_NAMES)
        means = list(stage.get("feature_mean", []))
        scales = list(stage.get("feature_scale", []))
        weights = list(stage.get("weights", []))
        if len(means) != n_features or len(scales) != n_features or len(weights) != n_features:
            raise ValueError(f"{stage_name} stage feature dimensions are invalid")
        try:
            float(stage.get("bias", 0.0))
            threshold = float(stage.get("minimum_probability", 0.55))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{stage_name} stage scalar fields are invalid") from exc
        if not 0.5 <= threshold <= 1.0:
            raise ValueError(f"{stage_name} minimum_probability must be in [0.5, 1.0]")

    def _stage_positive_probability(
        self,
        stage_name: str,
        features: dict[str, float],
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
        logit = sum(weight * value for weight, value in zip(weights, vector)) + float(
            stage["bias"]
        )
        if logit >= 0.0:
            exp_neg = math.exp(-logit)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(logit)
        return exp_pos / (1.0 + exp_pos)

    def classify(self, *, frame_state: Any, p2: dict[str, Any]) -> dict[str, Any]:
        features = self.features(frame_state=frame_state, p2=p2)
        flight = dict(dict(p2.get("possession", {}) or {}).get("flight", {}) or {})
        flight_active = bool(flight.get("active", False))

        airborne_probability = self._stage_positive_probability("airborne", features)
        type_shot_probability = self._stage_positive_probability("type", features)
        label = "unresolved"
        confidence = 0.0
        decision_stage = "no_model"

        stage_probabilities: dict[str, Any] | None = None
        output_probabilities: dict[str, float] | None = None
        if airborne_probability is not None and type_shot_probability is not None:
            stage1 = dict(self.model["stages"]["airborne"])
            stage2 = dict(self.model["stages"]["type"])
            airborne_minimum = self._float(stage1.get("minimum_probability"), 0.55)
            type_minimum = self._float(stage2.get("minimum_probability"), 0.55)
            pass_probability = 1.0 - type_shot_probability
            stage_probabilities = {
                "airborne": {
                    "continuity": round(1.0 - airborne_probability, 6),
                    "airborne": round(airborne_probability, 6),
                    "minimum_probability": round(airborne_minimum, 6),
                },
                "type": {
                    "pass_flight": round(pass_probability, 6),
                    "shot_flight": round(type_shot_probability, 6),
                    "minimum_probability": round(type_minimum, 6),
                },
            }
            output_probabilities = {
                "unresolved": 1.0 - airborne_probability,
                "pass_flight": airborne_probability * pass_probability,
                "shot_flight": airborne_probability * type_shot_probability,
            }

            if not flight_active:
                decision_stage = "not_active_flight"
            elif airborne_probability < airborne_minimum:
                confidence = 1.0 - airborne_probability
                decision_stage = "stage1_continuity"
            else:
                type_confidence = max(pass_probability, type_shot_probability)
                if type_confidence < type_minimum:
                    confidence = airborne_probability * type_confidence
                    decision_stage = "stage2_abstain"
                elif type_shot_probability >= 0.5:
                    label = "shot_flight"
                    confidence = airborne_probability * type_shot_probability
                    decision_stage = "stage2_shot"
                else:
                    label = "pass_flight"
                    confidence = airborne_probability * pass_probability
                    decision_stage = "stage2_pass"

        return {
            "version": self.VERSION,
            "model_loaded": self.model_loaded,
            "model_path": None if self.model_path is None else str(self.model_path),
            "flight_active": flight_active,
            "label": label,
            "confidence": round(float(confidence), 6),
            "decision_stage": decision_stage,
            "stage_probabilities": stage_probabilities,
            "probabilities": None
            if output_probabilities is None
            else {
                key: round(float(value), 6)
                for key, value in output_probabilities.items()
            },
            "features": {key: round(float(value), 6) for key, value in features.items()},
        }
