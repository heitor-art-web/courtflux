from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


MODEL_NAME = "vit_base_patch8_224.augreg2_in21k_ft_in1k"
TARGET_SIZE = (224, 224)


class TiedDigitAwareClassifier(torch.nn.Module):
    """Classifier contract used by the uncertainty-jnr ViT-B checkpoint.

    This is the runtime form of the checkpoint contract benchmarked in PR #37.
    It emits logits for jersey numbers 0..99 plus an explicit absent logit.
    """

    def __init__(
        self,
        embed_dim: int,
        embedding_type: str = "multiplicative",
        per_digit_bias: bool = False,
    ) -> None:
        super().__init__()
        if embedding_type not in {"additive", "multiplicative"}:
            raise ValueError(f"Unsupported embedding_type={embedding_type}")
        self.embedding_type = embedding_type
        self.per_digit_bias = bool(per_digit_bias)
        self.digit_classifier = torch.nn.Linear(embed_dim, 10, bias=False)
        self.absent_classifier = torch.nn.Linear(embed_dim, 1)
        bias_width = 10 if self.per_digit_bias else 1
        self.position_biases = torch.nn.Parameter(torch.zeros(3, bias_width))
        initial = (
            torch.randn(3, embed_dim)
            if embedding_type == "additive"
            else torch.ones(3, embed_dim)
        )
        self.position_embeddings = torch.nn.Parameter(initial)
        self.register_buffer(
            "tens_idx",
            torch.div(torch.arange(10, 100), 10, rounding_mode="floor"),
        )
        self.register_buffer("ones_idx", torch.remainder(torch.arange(10, 100), 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding_type == "additive":
            positioned = x.unsqueeze(1) + self.position_embeddings.unsqueeze(0)
        else:
            positioned = x.unsqueeze(1) * self.position_embeddings.unsqueeze(0)
        digit_logits = self.digit_classifier(positioned)
        digit_logits = digit_logits + self.position_biases.unsqueeze(0)
        single_logits = digit_logits[:, 0]
        tens_logits = digit_logits[:, 1]
        ones_logits = digit_logits[:, 2]
        absent_logit = self.absent_classifier(x)

        number_logits = torch.empty(
            x.shape[0], 100, device=x.device, dtype=x.dtype
        )
        number_logits[:, :10] = single_logits
        number_logits[:, 10:] = (
            tens_logits[:, self.tens_idx] + ones_logits[:, self.ones_idx]
        )
        return torch.cat([number_logits, absent_logit], dim=1)


class UncertaintyJNRModel(torch.nn.Module):
    """Runtime reproduction of the uncertainty-aware ViT-B benchmark model."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'timm'. Install requirements.txt; the ViT-B "
                "checkpoint contract requires timm==1.0.11."
            ) from exc

        self.timm_version = getattr(timm, "__version__", "unknown")
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False)
        embed_dim = int(self.backbone.embed_dim)
        self.time_embed = torch.nn.Sequential(torch.nn.Linear(1, embed_dim))
        with torch.no_grad():
            self.time_embed[0].weight.data.uniform_(-0.001, 0.001)
            self.time_embed[0].bias.data.zero_()
        self.classifier = TiedDigitAwareClassifier(
            embed_dim,
            embedding_type="multiplicative",
            per_digit_bias=False,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        all_logits = self.classifier(features)
        number_logits = all_logits[:, :100]
        alpha = torch.exp(number_logits) + 1.0
        concentration = alpha.sum(dim=1, keepdim=True)
        number_probs = alpha / concentration
        uncertainty = 100.0 / concentration.squeeze(1)
        return {
            "all_logits": all_logits,
            "number_logits": number_logits,
            "number_probs": number_probs,
            "uncertainty": uncertainty,
        }


class PerformanceUncertaintyJerseyAnalyzer(TeamOnlyAnalyzer):
    """P1 runtime jersey recognizer selected from the PR #37 benchmark path.

    Runtime contract:
    - CourtFlux `torso_direct` crops (same geometry used by the benchmark);
    - ViT-B uncertainty-aware 0..99 number classifier with explicit absent logit;
    - best-uncertainty temporal selection over a bounded best-crop buffer;
    - raw model number is resolved before roster validation;
    - roster can reject the raw number but never choose/substitute another number.

    The legacy tens/ones gate is intentionally not used here: this backend directly
    models 0..99 and therefore has no leading-zero/single-digit collapse convention.
    """

    BACKEND_NAME = "uncertainty_vit_b"
    DEFAULT_MIN_FRAMES = 2
    DEFAULT_MIN_SUPPORT = 2
    DEFAULT_MIN_VOTE_FRACTION = 0.50

    def __init__(
        self,
        profiling_path: str,
        device: str = "mps",
        model_path: str = "weights/jersey_uncertainty_vit_b.pt",
        sequence_length: int = 8,
    ) -> None:
        super().__init__(profiling_path=profiling_path)
        self.sequence_length = max(2, int(sequence_length))
        self.min_frames = self._env_int(
            "BASKETBALL_JERSEY_VIT_MIN_FRAMES", self.DEFAULT_MIN_FRAMES, 2, self.sequence_length
        )
        self.min_support = self._env_int(
            "BASKETBALL_JERSEY_VIT_MIN_SUPPORT", self.DEFAULT_MIN_SUPPORT, 2, self.sequence_length
        )
        self.min_vote_fraction = self._env_float(
            "BASKETBALL_JERSEY_VIT_MIN_VOTE_FRACTION",
            self.DEFAULT_MIN_VOTE_FRACTION,
            0.0,
            1.0,
        )
        self.uncertainty_max = self._env_optional_float(
            "BASKETBALL_JERSEY_VIT_UNCERTAINTY_MAX", 0.0, 1.0
        )
        self.pred_score_min = self._env_optional_float(
            "BASKETBALL_JERSEY_VIT_PRED_SCORE_MIN", 0.0, 1.0
        )
        self.best_crop_age_penalty = 0.006
        self.track_crop_buffers: dict[int, list[dict[str, Any]]] = {}
        self.track_observation_index: dict[int, int] = {}

        requested = str(device or "cpu").lower().strip()
        if requested == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif requested == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        checkpoint_path = Path(model_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Uncertainty ViT-B jersey checkpoint not found: {checkpoint_path}"
            )

        started = perf_counter()
        model = UncertaintyJNRModel()
        load = self._load_checkpoint(model, checkpoint_path)
        self.number_model = model.to(self.device).eval()
        self.model_path = checkpoint_path
        self.checkpoint_diagnostics = {
            "backend": self.BACKEND_NAME,
            "source": load["state_source"],
            "coverage": round(float(load["coverage"]), 6),
            "matched_keys": int(load["matched_keys"]),
            "checkpoint_tensor_keys": int(load["checkpoint_tensor_keys"]),
            "missing_keys": list(load["missing_keys"]),
            "unexpected_keys": list(load["unexpected_keys"]),
            "architecture": "uncertainty_vit_b_tied_digit_aware_v1",
            "model_name": MODEL_NAME,
            "timm_version": str(self.number_model.timm_version),
            "img_size": 224,
            "number_classes": 100,
            "absent_class": True,
            "preprocessing": "torso_direct_bicubic_224_rgb_minus1_plus1",
            "temporal_aggregation": "best_uncertainty_over_best_crop_buffer",
            "roster_policy": "validator_only",
            "policy": {
                "require_not_absent": True,
                "min_frames": self.min_frames,
                "min_support": self.min_support,
                "min_vote_fraction": self.min_vote_fraction,
                "uncertainty_max": self.uncertainty_max,
                "pred_score_min": self.pred_score_min,
            },
            "historical_direction": "PR37 selected ViT-B torso_direct best_uncertainty min_frames=2",
        }
        print(
            "🪪 Jersey backend: uncertainty_vit_b "
            f"checkpoint={checkpoint_path.name} device={self.device.type} "
            f"coverage={load['coverage']:.2%} timm={self.number_model.timm_version} "
            f"loaded_in={perf_counter() - started:.2f}s",
            flush=True,
        )

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        raw = os.getenv(name)
        try:
            value = int(raw) if raw is not None and raw.strip() else int(default)
        except (TypeError, ValueError):
            value = int(default)
        return max(int(minimum), min(int(maximum), value))

    @staticmethod
    def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
        raw = os.getenv(name)
        try:
            value = float(raw) if raw is not None and raw.strip() else float(default)
        except (TypeError, ValueError):
            value = float(default)
        return max(float(minimum), min(float(maximum), value))

    @staticmethod
    def _env_optional_float(name: str, minimum: float, maximum: float) -> float | None:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return max(float(minimum), min(float(maximum), value))

    @staticmethod
    def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Expected uncertainty-jnr checkpoint with top-level 'model_state_dict'."
            )
        raw_state = checkpoint["model_state_dict"]
        if not isinstance(raw_state, dict):
            raise RuntimeError("'model_state_dict' is not a state dictionary.")

        state = {
            str(key).replace("._orig_mod.", "."): value
            for key, value in raw_state.items()
        }
        model_state = model.state_dict()
        shared = set(model_state).intersection(state)
        mismatches = [
            (key, tuple(state[key].shape), tuple(model_state[key].shape))
            for key in shared
            if tuple(state[key].shape) != tuple(model_state[key].shape)
        ]
        if mismatches:
            raise RuntimeError(f"ViT jersey checkpoint/model shape mismatch: {mismatches[:8]}")

        missing, unexpected = model.load_state_dict(state, strict=False)
        checkpoint_numel = sum(
            int(value.numel()) for value in state.values() if torch.is_tensor(value)
        )
        matched_numel = sum(
            int(state[key].numel())
            for key in shared
            if torch.is_tensor(state[key])
        )
        coverage = matched_numel / checkpoint_numel if checkpoint_numel else 0.0
        if coverage < 0.995:
            raise RuntimeError(
                "ViT jersey checkpoint coverage is too low for runtime promotion: "
                f"{coverage:.2%}; missing={list(missing)[:8]}; "
                f"unexpected={list(unexpected)[:8]}"
            )
        return {
            "state_source": "root.model_state_dict",
            "checkpoint_tensor_keys": len(state),
            "matched_keys": len(shared),
            "coverage": coverage,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
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

    @staticmethod
    def _torso_direct(player_crop: np.ndarray) -> np.ndarray | None:
        if player_crop is None or player_crop.size == 0:
            return None
        height, width = player_crop.shape[:2]
        if height < 40 or width < 25:
            return None
        y1, y2 = int(height * 0.15), max(int(height * 0.65), int(height * 0.15) + 1)
        x1, x2 = int(width * 0.10), max(int(width * 0.90), int(width * 0.10) + 1)
        torso = player_crop[y1:y2, x1:x2]
        return torso if torso.size else None

    def observe_track(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_id: int | None = None,
    ) -> None:
        if track_id < 0:
            return
        torso = self._torso_direct(crop)
        if torso is None:
            return
        ordinal = int(self.track_observation_index.get(track_id, 0)) + 1
        self.track_observation_index[track_id] = ordinal
        item = {
            "crop": torso.copy(),
            "quality": self._crop_quality(torso),
            "ordinal": ordinal,
            "frame_id": frame_id,
        }
        ranked = list(self.track_crop_buffers.get(track_id, [])) + [item]
        ranked.sort(
            key=lambda row: (
                float(row.get("quality", 0.0))
                - self.best_crop_age_penalty
                * max(0, ordinal - int(row.get("ordinal", ordinal)))
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
        return len(self.track_crop_buffers.get(track_id, [])) >= self.min_frames

    @staticmethod
    def _preprocess_bgr(crop: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (TARGET_SIZE[1], TARGET_SIZE[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).float()
        return (tensor / 127.5) - 1.0

    def _valid_roster_numbers(self, team_name: str | None) -> list[str]:
        if not team_name or team_name not in self.rosters:
            return []
        roster = (self.rosters.get(team_name, {}) or {}).get("roster", {}) or {}
        values = []
        for key in roster:
            text = str(key).strip()
            if text.startswith("?") or not text.isdigit():
                continue
            number = int(text)
            if 0 <= number <= 99:
                values.append(str(number))
        return sorted(set(values), key=lambda value: int(value))

    @staticmethod
    def _top_numbers(probabilities: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
        order = np.argsort(probabilities)[::-1][: max(1, int(limit))]
        return [
            {"number": str(int(index)), "posterior": round(float(probabilities[index]), 6)}
            for index in order
        ]

    @staticmethod
    def _digit_marginals(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Diagnostic tens/ones marginals derived from the raw 0..99 distribution.

        Index 0 in the tens marginal means a single-digit number. This preserves the
        debug tooling contract without changing how the ViT backend resolves numbers.
        """
        tens = np.zeros(10, dtype=np.float64)
        ones = np.zeros(10, dtype=np.float64)
        for number, probability in enumerate(probabilities[:100]):
            value = float(probability)
            tens[0 if number <= 9 else number // 10] += value
            ones[number % 10] += value
        return tens, ones

    @staticmethod
    def _top_digits(probabilities: np.ndarray, limit: int = 3) -> list[dict[str, Any]]:
        order = np.argsort(probabilities)[::-1][: max(1, int(limit))]
        return [
            {"digit": int(index), "probability": round(float(probabilities[index]), 6)}
            for index in order
        ]

    def predict_jersey(self, track_id: int, team_name: str | None = None) -> dict[str, Any]:
        if track_id < 0 or not self.ready(track_id):
            return {
                "accepted": False,
                "rejection_reason": "sequence_not_ready",
                "roster_policy": "validator_only",
            }

        buffered = list(self.track_crop_buffers.get(track_id, []))[: self.sequence_length]
        tensors = [self._preprocess_bgr(item["crop"]) for item in buffered]
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            output = self.number_model(batch)

        probs = output["number_probs"].detach().float().cpu().numpy()
        uncertainties = output["uncertainty"].detach().float().cpu().numpy()
        all_logits = output["all_logits"].detach().float().cpu().numpy()

        per_crop: list[dict[str, Any]] = []
        eligible_indices: list[int] = []
        for index, (entry, row_probs) in enumerate(zip(buffered, probs)):
            order = np.argsort(row_probs)[::-1]
            pred = int(order[0])
            top2 = float(row_probs[order[1]]) if len(order) > 1 else 0.0
            max_number_logit = float(np.max(all_logits[index, :100]))
            absent_logit = float(all_logits[index, 100])
            absent = bool(int(np.argmax(all_logits[index])) == 100)
            score = float(row_probs[pred])
            uncertainty = float(uncertainties[index])
            tens, ones = self._digit_marginals(row_probs)
            row = {
                "frame_id": entry.get("frame_id"),
                "crop_quality": round(float(entry.get("quality", 0.0)), 6),
                "raw_canonical": str(pred),
                "raw_probability": round(score, 6),
                "pred_score": round(score, 6),
                "number_margin": round(max(0.0, score - top2), 6),
                "uncertainty": round(uncertainty, 6),
                "candidate_evidence": round(max(0.0, min(1.0, 1.0 - uncertainty)), 6),
                "absent_argmax": absent,
                "absent_margin": round(absent_logit - max_number_logit, 6),
                "number_top5": self._top_numbers(row_probs),
                "tens_top3": self._top_digits(tens),
                "ones_top3": self._top_digits(ones),
            }
            per_crop.append(row)

            if absent:
                continue
            if self.uncertainty_max is not None and uncertainty > self.uncertainty_max:
                continue
            if self.pred_score_min is not None and score < self.pred_score_min:
                continue
            eligible_indices.append(index)

        valid = self._valid_roster_numbers(team_name)
        if len(eligible_indices) < self.min_frames:
            return self._result(
                accepted=False,
                reason="low_uncertainty_frame_support",
                team_name=team_name,
                candidate=None,
                posterior=0.0,
                margin=0.0,
                support=0,
                vote_fraction=0.0,
                uncertainty=1.0,
                evidence=0.0,
                valid=valid,
                roster_valid=False,
                buffered=buffered,
                per_crop=per_crop,
                aggregate_probs=None,
            )

        best_index = min(eligible_indices, key=lambda idx: float(uncertainties[idx]))
        aggregate_probs = probs[best_index].astype(np.float64)
        order = np.argsort(aggregate_probs)[::-1]
        candidate = str(int(order[0]))
        posterior = float(aggregate_probs[order[0]])
        second = float(aggregate_probs[order[1]]) if len(order) > 1 else 0.0
        margin = max(0.0, posterior - second)
        votes = [int(np.argmax(probs[idx])) for idx in eligible_indices]
        support = sum(1 for vote in votes if str(vote) == candidate)
        vote_fraction = support / max(len(eligible_indices), 1)
        selected_uncertainty = float(uncertainties[best_index])
        evidence = max(0.0, min(1.0, 1.0 - selected_uncertainty)) * vote_fraction
        roster_valid = candidate in valid

        if support < self.min_support:
            accepted, reason = False, "low_raw_consensus_support"
        elif vote_fraction < self.min_vote_fraction:
            accepted, reason = False, "low_raw_consensus_vote_fraction"
        elif not team_name:
            accepted, reason = False, "team_unknown_for_roster_validation"
        elif not valid:
            accepted, reason = False, "no_known_roster_numbers"
        elif not roster_valid:
            accepted, reason = False, "raw_not_in_roster"
        else:
            accepted, reason = True, "accepted"

        return self._result(
            accepted=accepted,
            reason=reason,
            team_name=team_name,
            candidate=candidate,
            posterior=posterior,
            margin=margin,
            support=support,
            vote_fraction=vote_fraction,
            uncertainty=selected_uncertainty,
            evidence=evidence,
            valid=valid,
            roster_valid=roster_valid,
            buffered=buffered,
            per_crop=per_crop,
            aggregate_probs=aggregate_probs,
        )

    def _result(
        self,
        *,
        accepted: bool,
        reason: str,
        team_name: str | None,
        candidate: str | None,
        posterior: float,
        margin: float,
        support: int,
        vote_fraction: float,
        uncertainty: float,
        evidence: float,
        valid: list[str],
        roster_valid: bool,
        buffered: list[dict[str, Any]],
        per_crop: list[dict[str, Any]],
        aggregate_probs: np.ndarray | None,
    ) -> dict[str, Any]:
        if aggregate_probs is None:
            top_candidates: list[dict[str, Any]] = []
            tens_top3: list[dict[str, Any]] = []
            ones_top3: list[dict[str, Any]] = []
        else:
            top_candidates = self._top_numbers(aggregate_probs)
            tens, ones = self._digit_marginals(aggregate_probs)
            tens_top3 = self._top_digits(tens)
            ones_top3 = self._top_digits(ones)

        cross_team_validation = {}
        for candidate_team in self.rosters:
            team_valid = self._valid_roster_numbers(str(candidate_team))
            is_valid = bool(candidate is not None and candidate in team_valid)
            cross_team_validation[str(candidate_team)] = {
                "candidate_number": candidate if is_valid else None,
                "candidate_posterior": round(posterior, 6) if is_valid else 0.0,
                "candidate_margin": round(margin, 6) if is_valid else 0.0,
                "roster_valid": is_valid,
            }

        return {
            "accepted": bool(accepted),
            "rejection_reason": str(reason),
            "team": team_name,
            "candidate_number": candidate,
            "raw_number": candidate,
            "raw_canonical": candidate,
            "raw_decode_reason": "uncertainty_vit_b_best_uncertainty_raw_0_99",
            "candidate_probability": round(float(posterior), 6),
            "candidate_posterior": round(float(posterior), 6),
            "candidate_margin": round(float(margin), 6),
            "candidate_count": int(support),
            "candidate_vote_fraction": round(float(vote_fraction), 6),
            "candidate_uncertainty": round(float(uncertainty), 6),
            "candidate_evidence": round(float(evidence), 6),
            "top_candidates": top_candidates,
            "tens_top3": tens_top3,
            "ones_top3": ones_top3,
            "single_digit_gate": {
                "required": False,
                "passed": True,
                "reason": "not_applicable_full_0_99_classifier",
            },
            "valid_roster_numbers": valid,
            "roster_valid": bool(roster_valid),
            "roster_policy": "validator_only",
            "cross_team_validation": cross_team_validation,
            "cross_team_best": cross_team_validation,
            "sequence_crops": len(buffered),
            "eligible_uncertainty_crops": sum(
                1
                for row in per_crop
                if not bool(row.get("absent_argmax"))
                and (self.uncertainty_max is None or float(row.get("uncertainty", 1.0)) <= self.uncertainty_max)
                and (self.pred_score_min is None or float(row.get("pred_score", 0.0)) >= self.pred_score_min)
            ),
            "buffer_policy": "best_crop_with_age_penalty",
            "aggregation": "best_uncertainty",
            "crop_quality": [round(float(item.get("quality", 0.0)), 6) for item in buffered],
            "per_crop_predictions": per_crop,
            "checkpoint": self.checkpoint_diagnostics,
            "p1_raw_roster_consistency_gate": True,
        }
