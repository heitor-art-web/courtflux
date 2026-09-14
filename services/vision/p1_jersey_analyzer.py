from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from services.vision.performance_jersey_analyzer import (
    PerformanceJerseyAnalyzer as BasePerformanceJerseyAnalyzer,
)


class PerformanceJerseyAnalyzer(BasePerformanceJerseyAnalyzer):
    """P1 Identity v2 Jersey recognizer.

    The checkpoint remains the existing single-frame network. Identity v2 keeps the
    best torso observations, aggregates the full tens/ones probability distributions
    across those crops, and only then resolves a raw jersey number. A conservative
    single-digit gate prevents an ambiguous tens head from collapsing a two-digit
    jersey into #4/#6/etc. The roster may reject that raw result, but it can never
    choose or substitute a number.
    """

    SINGLE_DIGIT_TENS_MIN_PROBABILITY = 0.80
    SINGLE_DIGIT_TENS_MIN_MARGIN = 0.35
    SINGLE_DIGIT_TENS_MIN_SUPPORT_RATIO = 0.75

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.track_crop_buffers: dict[int, list[dict[str, Any]]] = {}
        self.track_observation_index: dict[int, int] = {}
        self.best_crop_age_penalty = 0.006
        self.checkpoint_diagnostics = dict(self.checkpoint_diagnostics)
        self.checkpoint_diagnostics["temporal_aggregation"] = (
            f"best_{self.sequence_length}_crop_digit_probability_consensus_v3"
        )
        self.checkpoint_diagnostics["roster_policy"] = "validator_only"
        self.checkpoint_diagnostics["single_digit_gate"] = {
            "tens_probability_min": self.SINGLE_DIGIT_TENS_MIN_PROBABILITY,
            "tens_margin_min": self.SINGLE_DIGIT_TENS_MIN_MARGIN,
            "crop_support_ratio_min": self.SINGLE_DIGIT_TENS_MIN_SUPPORT_RATIO,
        }

    @staticmethod
    def _crop_quality(crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        area = float(gray.shape[0] * gray.shape[1])
        if area <= 0:
            return 0.0
        sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 420.0, 1.0)
        contrast = min(float(np.std(gray)) / 64.0, 1.0)
        exposure = float(np.mean((gray >= 32) & (gray <= 225)))
        resolution = min(area / float(90 * 55), 1.0)
        return float(
            0.36 * sharpness
            + 0.24 * contrast
            + 0.24 * exposure
            + 0.16 * resolution
        )

    def observe_track(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_id: int | None = None,
    ) -> None:
        if track_id < 0 or crop is None or crop.size == 0:
            return
        height, width = crop.shape[:2]
        if height < 40 or width < 25:
            return

        ymin, ymax = int(height * 0.15), int(height * 0.65)
        xmin, xmax = int(width * 0.10), int(width * 0.90)
        torso = crop[ymin:ymax, xmin:xmax]
        if torso.size == 0:
            return

        ordinal = int(self.track_observation_index.get(track_id, 0)) + 1
        self.track_observation_index[track_id] = ordinal
        quality = self._crop_quality(torso)
        buffer = self.track_crop_buffers.setdefault(track_id, [])
        candidate = {
            "crop": torso.copy(),
            "quality": quality,
            "ordinal": ordinal,
            "frame_id": frame_id,
        }
        ranked = list(buffer) + [candidate]
        ranked.sort(
            key=lambda item: (
                float(item.get("quality", 0.0))
                - self.best_crop_age_penalty
                * max(0, ordinal - int(item.get("ordinal", ordinal)))
            ),
            reverse=True,
        )
        self.track_crop_buffers[track_id] = ranked[: self.sequence_length]

        if len(self.track_crop_buffers) > 250:
            keep = sorted(
                self.track_crop_buffers,
                key=lambda key: int(self.track_observation_index.get(key, 0)),
                reverse=True,
            )[:100]
            self.track_crop_buffers = {key: self.track_crop_buffers[key] for key in keep}
            self.track_observation_index = {
                key: self.track_observation_index.get(key, 0) for key in keep
            }

    def ready(self, track_id: int) -> bool:
        return len(self.track_crop_buffers.get(track_id, [])) >= self.sequence_length

    @staticmethod
    def _raw_consensus(
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Legacy winner-vote helper kept for tests/backward diagnostics."""
        votes: dict[str, float] = {}
        counts: dict[str, int] = {}
        probabilities: dict[str, list[float]] = {}
        for row in rows:
            number = row.get("raw_canonical")
            if number is None:
                continue
            number = str(number)
            probability = max(0.0, float(row.get("raw_probability", 0.0) or 0.0))
            quality = max(0.0, min(float(row.get("crop_quality", 0.0) or 0.0), 1.0))
            weight = max(1e-6, probability) * (0.65 + 0.35 * quality)
            votes[number] = votes.get(number, 0.0) + weight
            counts[number] = counts.get(number, 0) + 1
            probabilities.setdefault(number, []).append(probability)

        if not votes:
            return {
                "candidate_number": None,
                "candidate_probability": 0.0,
                "candidate_posterior": 0.0,
                "candidate_margin": 0.0,
                "candidate_count": 0,
                "top_candidates": [],
            }

        ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        winner, winner_weight = ordered[0]
        second_weight = ordered[1][1] if len(ordered) > 1 else 0.0
        total = sum(votes.values()) or 1e-12
        winner_probs = probabilities.get(winner, [])
        candidate_probability = (
            sum(winner_probs) / max(len(winner_probs), 1) if winner_probs else 0.0
        )
        return {
            "candidate_number": winner,
            "candidate_probability": round(candidate_probability, 6),
            "candidate_posterior": round(winner_weight / total, 6),
            "candidate_margin": round(max(0.0, (winner_weight - second_weight) / total), 6),
            "candidate_count": int(counts.get(winner, 0)),
            "top_candidates": [
                {
                    "number": number,
                    "posterior": round(weight / total, 6),
                    "count": int(counts.get(number, 0)),
                }
                for number, weight in ordered[:3]
            ],
        }

    @staticmethod
    def _weighted_digit_distribution(
        rows: list[dict[str, Any]],
        key: str,
    ) -> list[float]:
        class_count = max((len(list(row.get(key) or [])) for row in rows), default=0)
        if class_count <= 0:
            return []
        totals = [0.0] * class_count
        total_weight = 0.0
        for row in rows:
            probs = [max(0.0, float(value)) for value in list(row.get(key) or [])]
            if not probs:
                continue
            quality = max(0.0, min(float(row.get("crop_quality", 0.0) or 0.0), 1.0))
            weight = 0.65 + 0.35 * quality
            for index, probability in enumerate(probs[:class_count]):
                totals[index] += weight * probability
            total_weight += weight
        if total_weight <= 1e-12:
            return []
        averaged = [value / total_weight for value in totals]
        normalizer = sum(averaged) or 1e-12
        return [value / normalizer for value in averaged]

    @staticmethod
    def _canonical_digit_pair(
        tens_index: int,
        ones_index: int,
        tens_class_count: int,
    ) -> str | None:
        if ones_index < 0 or ones_index >= 10:
            return None
        if tens_class_count > 10 and tens_index == 10:
            return str(ones_index)
        if tens_index < 0 or tens_index >= 10:
            return None
        if tens_class_count <= 10 and tens_index == 0:
            return str(ones_index)
        return str(int(f"{tens_index}{ones_index}"))

    @staticmethod
    def _top_digits(probabilities: list[float], limit: int = 3) -> list[dict[str, Any]]:
        ordered = sorted(
            enumerate(probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"digit": int(index), "probability": round(float(probability), 6)}
            for index, probability in ordered[:limit]
        ]

    @classmethod
    def _probabilistic_digit_consensus(
        cls,
        rows: list[dict[str, Any]],
        *,
        single_digit_min_probability: float | None = None,
        single_digit_min_margin: float | None = None,
        single_digit_min_support_ratio: float | None = None,
    ) -> dict[str, Any]:
        """Fuse full digit distributions before resolving the jersey number."""
        min_probability = (
            cls.SINGLE_DIGIT_TENS_MIN_PROBABILITY
            if single_digit_min_probability is None
            else float(single_digit_min_probability)
        )
        min_margin = (
            cls.SINGLE_DIGIT_TENS_MIN_MARGIN
            if single_digit_min_margin is None
            else float(single_digit_min_margin)
        )
        min_support_ratio = (
            cls.SINGLE_DIGIT_TENS_MIN_SUPPORT_RATIO
            if single_digit_min_support_ratio is None
            else float(single_digit_min_support_ratio)
        )

        tens = cls._weighted_digit_distribution(rows, "tens_probs")
        ones = cls._weighted_digit_distribution(rows, "ones_probs")
        if not tens or not ones:
            return {
                "candidate_number": None,
                "candidate_probability": 0.0,
                "candidate_posterior": 0.0,
                "candidate_margin": 0.0,
                "candidate_count": 0,
                "top_candidates": [],
                "tens_top3": cls._top_digits(tens),
                "ones_top3": cls._top_digits(ones),
                "single_digit_gate": {
                    "required": False,
                    "passed": False,
                    "reason": "missing_digit_probabilities",
                },
            }

        pair_scores: dict[str, float] = {}
        for tens_index, tens_probability in enumerate(tens):
            for ones_index, ones_probability in enumerate(ones):
                number = cls._canonical_digit_pair(
                    tens_index,
                    ones_index,
                    len(tens),
                )
                if number is None:
                    continue
                pair_scores[number] = pair_scores.get(number, 0.0) + (
                    tens_probability * ones_probability
                )

        ordered = sorted(pair_scores.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return {
                "candidate_number": None,
                "candidate_probability": 0.0,
                "candidate_posterior": 0.0,
                "candidate_margin": 0.0,
                "candidate_count": 0,
                "top_candidates": [],
                "tens_top3": cls._top_digits(tens),
                "ones_top3": cls._top_digits(ones),
                "single_digit_gate": {
                    "required": False,
                    "passed": False,
                    "reason": "no_valid_digit_pair",
                },
            }

        total = sum(pair_scores.values()) or 1e-12
        winner, winner_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0.0

        support = 0
        for row in rows:
            row_tens = list(row.get("tens_probs") or [])
            row_ones = list(row.get("ones_probs") or [])
            if not row_tens or not row_ones:
                continue
            row_number = cls._canonical_digit_pair(
                int(np.argmax(row_tens)),
                int(np.argmax(row_ones)),
                len(row_tens),
            )
            if row_number == winner:
                support += 1

        try:
            single_digit = 0 <= int(winner) <= 9
        except (TypeError, ValueError):
            single_digit = False

        single_index = 10 if len(tens) > 10 else 0
        single_probability = (
            float(tens[single_index]) if single_index < len(tens) else 0.0
        )
        non_single = [
            (index, float(tens[index]))
            for index in range(1, min(10, len(tens)))
        ]
        strongest_non_single_digit, strongest_non_single_probability = (
            max(non_single, key=lambda item: item[1])
            if non_single
            else (-1, 0.0)
        )
        single_margin = max(
            0.0,
            single_probability - strongest_non_single_probability,
        )

        tens_support_count = 0
        tens_row_count = 0
        for row in rows:
            row_tens = list(row.get("tens_probs") or [])
            if not row_tens or single_index >= len(row_tens):
                continue
            tens_row_count += 1
            if int(np.argmax(row_tens)) == single_index:
                tens_support_count += 1
        single_support_ratio = (
            tens_support_count / tens_row_count if tens_row_count else 0.0
        )

        single_gate_passed = bool(
            not single_digit
            or (
                single_probability >= min_probability
                and single_margin >= min_margin
                and single_support_ratio >= min_support_ratio
            )
        )
        if not single_digit:
            gate_reason = "not_required_two_digit_candidate"
        elif single_gate_passed:
            gate_reason = "strong_single_digit_tens_evidence"
        else:
            gate_reason = "ambiguous_single_digit_tens"

        return {
            "candidate_number": winner,
            "candidate_probability": round(float(winner_score), 6),
            "candidate_posterior": round(float(winner_score / total), 6),
            "candidate_margin": round(
                max(0.0, float((winner_score - second_score) / total)),
                6,
            ),
            "candidate_count": int(support),
            "top_candidates": [
                {
                    "number": number,
                    "posterior": round(float(score / total), 6),
                }
                for number, score in ordered[:5]
            ],
            "tens_top3": cls._top_digits(tens),
            "ones_top3": cls._top_digits(ones),
            "single_digit_gate": {
                "required": bool(single_digit),
                "passed": bool(single_gate_passed),
                "reason": gate_reason,
                "tens_single_probability": round(single_probability, 6),
                "tens_single_margin": round(single_margin, 6),
                "crop_support_ratio": round(single_support_ratio, 6),
                "crop_support_count": int(tens_support_count),
                "crop_count": int(tens_row_count),
                "strongest_non_single_digit": int(strongest_non_single_digit),
                "strongest_non_single_probability": round(
                    strongest_non_single_probability,
                    6,
                ),
                "thresholds": {
                    "tens_probability_min": round(min_probability, 6),
                    "tens_margin_min": round(min_margin, 6),
                    "crop_support_ratio_min": round(min_support_ratio, 6),
                },
            },
        }

    @staticmethod
    def _legacy_consistency_gate(prediction: dict[str, Any]) -> dict[str, Any]:
        prediction["p1_raw_roster_consistency_gate"] = True
        if not prediction.get("accepted"):
            return prediction
        candidate = prediction.get("candidate_number")
        raw_canonical = prediction.get("raw_canonical")
        if (
            candidate is not None
            and raw_canonical is not None
            and str(candidate) != str(raw_canonical)
        ):
            prediction["accepted"] = False
            prediction["rejection_reason"] = "raw_roster_disagreement"
            prediction["rejected_candidate_number"] = str(candidate)
        return prediction

    def predict_jersey(self, track_id: int, team_name: Optional[str] = None) -> dict:
        # Some unit tests construct this wrapper with __new__ and monkeypatch the legacy
        # parent prediction. Keep that path intact while runtime instances use Identity v2.
        if not hasattr(self, "number_model") or not hasattr(self, "track_observation_index"):
            legacy = dict(
                super().predict_jersey(track_id=track_id, team_name=team_name) or {}
            )
            return self._legacy_consistency_gate(legacy)

        if track_id < 0 or not self.ready(track_id):
            return {
                "accepted": False,
                "rejection_reason": "sequence_not_ready",
                "roster_policy": "validator_only",
            }

        buffered = list(self.track_crop_buffers.get(track_id, []))[: self.sequence_length]
        tensors = []
        for entry in buffered:
            frame_crop = entry["crop"]
            rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))

        input_tensor = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            tens_logits, ones_logits = self.number_model(input_tensor)
            tens_rows = torch.softmax(tens_logits, dim=1).detach().cpu().tolist()
            ones_rows = torch.softmax(ones_logits, dim=1).detach().cpu().tolist()

        per_crop: list[dict[str, Any]] = []
        probability_rows: list[dict[str, Any]] = []
        for entry, tens_probs, ones_probs in zip(buffered, tens_rows, ones_rows):
            raw = dict(self._raw_prediction(tens_probs, ones_probs))
            quality = round(float(entry.get("quality", 0.0)), 6)
            raw["crop_quality"] = quality
            raw["frame_id"] = entry.get("frame_id")
            raw["tens_top3"] = self._top_digits([float(v) for v in tens_probs])
            raw["ones_top3"] = self._top_digits([float(v) for v in ones_probs])
            per_crop.append(raw)
            probability_rows.append(
                {
                    "raw_canonical": raw.get("raw_canonical"),
                    "tens_probs": [float(v) for v in tens_probs],
                    "ones_probs": [float(v) for v in ones_probs],
                    "crop_quality": quality,
                }
            )

        consensus = self._probabilistic_digit_consensus(probability_rows)
        candidate = consensus.get("candidate_number")
        posterior = float(consensus.get("candidate_posterior", 0.0) or 0.0)
        margin = float(consensus.get("candidate_margin", 0.0) or 0.0)
        support = int(consensus.get("candidate_count", 0) or 0)
        single_digit_gate = dict(consensus.get("single_digit_gate", {}) or {})

        valid = self._valid_roster_numbers(team_name)
        roster_valid = bool(candidate is not None and str(candidate) in valid)
        if candidate is None:
            accepted = False
            reason = "no_raw_consensus"
        elif support < 2:
            accepted = False
            reason = "low_raw_consensus_support"
        elif single_digit_gate.get("required") and not single_digit_gate.get("passed"):
            accepted = False
            reason = "ambiguous_single_digit_tens"
        elif posterior < 0.50:
            accepted = False
            reason = "low_raw_consensus_posterior"
        elif margin < 0.12:
            accepted = False
            reason = "low_raw_consensus_margin"
        elif not team_name:
            accepted = False
            reason = "team_unknown_for_roster_validation"
        elif not valid:
            accepted = False
            reason = "no_known_roster_numbers"
        elif not roster_valid:
            accepted = False
            reason = "raw_not_in_roster"
        else:
            accepted = True
            reason = "accepted"

        cross_team_validation = {}
        for candidate_team in self.rosters:
            team_valid = self._valid_roster_numbers(str(candidate_team))
            is_valid = bool(candidate is not None and str(candidate) in team_valid)
            cross_team_validation[str(candidate_team)] = {
                "candidate_number": str(candidate) if is_valid else None,
                "candidate_posterior": round(posterior, 6) if is_valid else 0.0,
                "candidate_margin": round(margin, 6) if is_valid else 0.0,
                "roster_valid": is_valid,
            }

        result = {
            "accepted": accepted,
            "rejection_reason": reason,
            "team": team_name,
            "raw_number": candidate,
            "raw_canonical": candidate,
            "raw_decode_reason": "digit_probability_temporal_consensus",
            "raw_probability": consensus.get("candidate_probability", 0.0),
            **consensus,
            "valid_roster_numbers": valid,
            "roster_valid": roster_valid,
            "roster_policy": "validator_only",
            "cross_team_validation": cross_team_validation,
            # Compatibility key: values are validation of the same raw consensus, never
            # a team-conditioned alternate decode.
            "cross_team_best": cross_team_validation,
            "sequence_crops": len(buffered),
            "buffer_policy": "best_crop_with_age_penalty",
            "crop_quality": [
                round(float(item.get("quality", 0.0)), 6) for item in buffered
            ],
            "per_crop_predictions": per_crop,
            "checkpoint": self.checkpoint_diagnostics,
            "p1_raw_roster_consistency_gate": True,
        }
        return result
