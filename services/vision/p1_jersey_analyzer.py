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

    The checkpoint remains the existing single-frame network. Identity v2 changes only
    temporal evidence handling: each track keeps the best torso observations (with a
    small age penalty), the network decodes each crop independently, and the final
    number comes from raw-number temporal consensus. The roster can reject that raw
    consensus, but it can never choose or substitute a number.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.track_crop_buffers: dict[int, list[dict[str, Any]]] = {}
        self.track_observation_index: dict[int, int] = {}
        self.best_crop_age_penalty = 0.006
        self.checkpoint_diagnostics = dict(self.checkpoint_diagnostics)
        self.checkpoint_diagnostics["temporal_aggregation"] = (
            f"best_{self.sequence_length}_crop_raw_number_consensus_v2"
        )
        self.checkpoint_diagnostics["roster_policy"] = "validator_only"

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
    def _legacy_consistency_gate(prediction: dict[str, Any]) -> dict[str, Any]:
        prediction["p1_raw_roster_consistency_gate"] = True
        if not prediction.get("accepted"):
            return prediction
        candidate = prediction.get("candidate_number")
        raw_canonical = prediction.get("raw_canonical")
        if candidate is not None and raw_canonical is not None and str(candidate) != str(raw_canonical):
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
        for entry, tens_probs, ones_probs in zip(buffered, tens_rows, ones_rows):
            raw = dict(self._raw_prediction(tens_probs, ones_probs))
            raw["crop_quality"] = round(float(entry.get("quality", 0.0)), 6)
            raw["frame_id"] = entry.get("frame_id")
            per_crop.append(raw)

        consensus = self._raw_consensus(per_crop)
        candidate = consensus.get("candidate_number")
        posterior = float(consensus.get("candidate_posterior", 0.0) or 0.0)
        margin = float(consensus.get("candidate_margin", 0.0) or 0.0)
        support = int(consensus.get("candidate_count", 0) or 0)

        valid = self._valid_roster_numbers(team_name)
        roster_valid = bool(candidate is not None and str(candidate) in valid)
        if candidate is None:
            accepted = False
            reason = "no_raw_consensus"
        elif support < 2:
            accepted = False
            reason = "low_raw_consensus_support"
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
            "raw_decode_reason": "best_crop_temporal_consensus",
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
            "crop_quality": [round(float(item.get("quality", 0.0)), 6) for item in buffered],
            "per_crop_predictions": per_crop,
            "checkpoint": self.checkpoint_diagnostics,
            "p1_raw_roster_consistency_gate": True,
        }
        return result
