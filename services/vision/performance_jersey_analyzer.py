from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


class CheckpointJerseyClassifier(nn.Module):
    """Single-frame jersey recognizer matching the repository-local checkpoint.

    The checkpoint discovered in production uses:
      backbone.features.*
      proj.*
      head_tens.*
      head_ones.*

    Temporal stability is intentionally handled outside this network by aggregating
    probabilities across real crops from the same tracked player.
    """

    def __init__(
        self,
        projection_in: int,
        projection_out: int,
        tens_classes: int,
        ones_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.proj = nn.Linear(projection_in, projection_out)
        self.head_tens = nn.Linear(projection_out, tens_classes)
        self.head_ones = nn.Linear(projection_out, ones_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        features = torch.relu(self.proj(features))
        return self.head_tens(features), self.head_ones(features)


class PerformanceJerseyAnalyzer(TeamOnlyAnalyzer):
    """Checkpoint-native jersey inference + temporal track-level aggregation."""

    CHECKPOINT_ARCH_VERSION = "single_frame_efficientnet_b0_proj_digit_heads_v1"

    def __init__(
        self,
        profiling_path: str,
        device: str = "mps",
        model_path: str = "weights/jersey_number_recon.pt",
        sequence_length: int = 8,
    ) -> None:
        super().__init__(profiling_path=profiling_path)
        self.sequence_length = max(2, int(sequence_length))
        self.track_crop_buffers: Dict[int, List[np.ndarray]] = {}

        requested = (device or "cpu").lower()
        if requested == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif requested == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Jersey checkpoint not found: {checkpoint_path}")

        started = perf_counter()
        print(
            f"📦 Loading jersey checkpoint [{self.CHECKPOINT_ARCH_VERSION}] "
            f"from {checkpoint_path} on {self.device.type}...",
            flush=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            state_source = "root.nn_module"
            state_dict = model.state_dict()
            model_state = state_dict
            img_size = int(getattr(model, "img_size", 128) or 128)
            projection_in = None
            projection_out = None
            tens_classes = None
            ones_classes = None
        else:
            state_source, state_dict = self._select_checkpoint_state(checkpoint)
            (
                projection_in,
                projection_out,
                tens_classes,
                ones_classes,
            ) = self._infer_head_dimensions(state_dict)

            model = CheckpointJerseyClassifier(
                projection_in=projection_in,
                projection_out=projection_out,
                tens_classes=tens_classes,
                ones_classes=ones_classes,
            )
            model_state = model.state_dict()
            self._validate_state_dict(state_dict, model_state, state_source)
            model.load_state_dict(state_dict, strict=True)
            img_size = int(self._find_scalar(checkpoint, ("img_size", "image_size", "input_size")) or 128)

        img_size = max(64, min(int(img_size), 512))
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.checkpoint_diagnostics = {
            "source": state_source,
            "coverage": 1.0,
            "shape_matches": int(len(model_state)),
            "key_matches": int(len(model_state)),
            "tensor_keys": int(len(state_dict)),
            "model_state_keys": int(len(model_state)),
            "missing_keys": 0,
            "unexpected_keys": 0,
            "parameter_count": int(parameter_count),
            "architecture": self.CHECKPOINT_ARCH_VERSION,
            "img_size": int(img_size),
            "projection_in": projection_in,
            "projection_out": projection_out,
            "tens_classes": tens_classes,
            "ones_classes": ones_classes,
            "temporal_aggregation": f"mean_probabilities_over_{self.sequence_length}_track_crops",
        }

        self.number_model = model.to(self.device).eval()
        print(
            f"✅ Jersey model ready in {perf_counter() - started:.2f}s; "
            f"checkpoint coverage=100.0%; state_keys={len(model_state)}; "
            f"imgsz={img_size}; heads={tens_classes}x{ones_classes}.",
            flush=True,
        )

    @staticmethod
    def _iter_tensor_dicts(
        value,
        path: str = "root",
        depth: int = 0,
    ) -> Iterable[tuple[str, dict]]:
        if depth > 5 or not isinstance(value, dict):
            return

        tensor_values = {
            str(key): item
            for key, item in value.items()
            if torch.is_tensor(item)
        }
        if tensor_values:
            yield path, tensor_values

        for key, item in value.items():
            if isinstance(item, dict):
                yield from PerformanceJerseyAnalyzer._iter_tensor_dicts(
                    item,
                    path=f"{path}.{key}",
                    depth=depth + 1,
                )

    @staticmethod
    def _state_variants(state_dict: dict) -> Iterable[tuple[str, dict]]:
        yield "identity", state_dict
        prefixes = ("module.", "model.", "number_model.", "network.", "net.")
        for prefix in prefixes:
            if any(str(key).startswith(prefix) for key in state_dict):
                yield (
                    f"strip:{prefix}",
                    {
                        (str(key)[len(prefix):] if str(key).startswith(prefix) else str(key)): value
                        for key, value in state_dict.items()
                    },
                )

    @classmethod
    def _select_checkpoint_state(cls, checkpoint) -> tuple[str, dict]:
        candidates = list(cls._iter_tensor_dicts(checkpoint)) if isinstance(checkpoint, dict) else []
        if not candidates:
            raise TypeError(
                "Unsupported jersey checkpoint format: no tensor state dictionary found."
            )

        required = {
            "proj.weight",
            "proj.bias",
            "head_tens.weight",
            "head_tens.bias",
            "head_ones.weight",
            "head_ones.bias",
        }
        best = None
        for source, raw_state in candidates:
            for variant_name, candidate in cls._state_variants(raw_state):
                required_matches = len(required.intersection(candidate.keys()))
                backbone_matches = sum(
                    1 for key in candidate if str(key).startswith("backbone.features.")
                )
                score = (
                    required_matches,
                    backbone_matches,
                    len(candidate),
                )
                if best is None or score > best[0]:
                    best = (
                        score,
                        f"{source}[{variant_name}]",
                        candidate,
                    )

        assert best is not None
        source = best[1]
        state_dict = best[2]
        missing_required = sorted(required.difference(state_dict.keys()))
        if missing_required:
            sample_keys = ", ".join(list(state_dict.keys())[-12:])
            raise RuntimeError(
                "Jersey checkpoint does not contain the checkpoint-native projection/digit heads. "
                f"Best tensor dictionary='{source}'. Missing keys={missing_required}. "
                f"Tail checkpoint keys: {sample_keys}"
            )

        return source, state_dict

    @staticmethod
    def _infer_head_dimensions(state_dict: dict) -> tuple[int, int, int, int]:
        proj_weight = state_dict["proj.weight"]
        tens_weight = state_dict["head_tens.weight"]
        ones_weight = state_dict["head_ones.weight"]

        if proj_weight.ndim != 2 or tens_weight.ndim != 2 or ones_weight.ndim != 2:
            raise RuntimeError(
                "Jersey checkpoint projection and digit-head weights must all be rank-2 tensors."
            )

        projection_out, projection_in = map(int, proj_weight.shape)
        tens_classes, tens_in = map(int, tens_weight.shape)
        ones_classes, ones_in = map(int, ones_weight.shape)

        if tens_in != projection_out or ones_in != projection_out:
            raise RuntimeError(
                "Jersey checkpoint head dimensions are inconsistent: "
                f"proj_out={projection_out}, head_tens_in={tens_in}, head_ones_in={ones_in}."
            )
        if projection_in != 1280:
            raise RuntimeError(
                "Jersey checkpoint projection does not match EfficientNet-B0 output: "
                f"expected 1280 input features, found {projection_in}."
            )
        if tens_classes < 10 or ones_classes < 10:
            raise RuntimeError(
                "Jersey checkpoint digit heads must expose at least digits 0-9: "
                f"tens_classes={tens_classes}, ones_classes={ones_classes}."
            )

        return projection_in, projection_out, tens_classes, ones_classes

    @staticmethod
    def _validate_state_dict(state_dict: dict, model_state: dict, source: str) -> None:
        missing = [key for key in model_state if key not in state_dict]
        unexpected = [key for key in state_dict if key not in model_state]
        shape_mismatches = [
            (
                key,
                tuple(state_dict[key].shape),
                tuple(model_state[key].shape),
            )
            for key in model_state
            if key in state_dict
            and torch.is_tensor(state_dict[key])
            and tuple(state_dict[key].shape) != tuple(model_state[key].shape)
        ]

        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "Jersey checkpoint still does not exactly match the inferred single-frame model. "
                f"source='{source}', missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"shape_mismatches={shape_mismatches[:5]}"
            )

        print(
            "🔎 Jersey checkpoint inspection: "
            f"source={source} shape_matches={len(model_state)}/{len(model_state)} "
            f"key_matches={len(model_state)} tensor_keys={len(state_dict)}",
            flush=True,
        )

    @staticmethod
    def _find_scalar(value, names: tuple[str, ...], depth: int = 0):
        if depth > 4 or not isinstance(value, dict):
            return None

        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, (int, float, str)):
                try:
                    return int(candidate)
                except (TypeError, ValueError):
                    pass

        for item in value.values():
            if isinstance(item, dict):
                found = PerformanceJerseyAnalyzer._find_scalar(
                    item,
                    names,
                    depth=depth + 1,
                )
                if found is not None:
                    return found
        return None

    def observe_track(self, crop: np.ndarray, track_id: int) -> None:
        """Collect one genuine torso observation per frame without neural inference."""
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

        buffer = self.track_crop_buffers.setdefault(track_id, [])
        buffer.append(torso.copy())
        if len(buffer) > self.sequence_length:
            del buffer[:-self.sequence_length]

        if len(self.track_crop_buffers) > 250:
            keep = list(self.track_crop_buffers.keys())[-100:]
            self.track_crop_buffers = {
                key: self.track_crop_buffers[key]
                for key in keep
            }

    def ready(self, track_id: int) -> bool:
        return len(self.track_crop_buffers.get(track_id, [])) >= self.sequence_length

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
    def _candidate_probability(
        number: str,
        tens_probs: list[float],
        ones_probs: list[float],
    ) -> float:
        value = int(number)
        ones_digit = value % 10
        if ones_digit >= len(ones_probs):
            return 0.0

        if value <= 9:
            # Some digit-wise checkpoints expose an explicit blank tens class at index
            # 10. For 10-class heads, legacy training represented one-digit numbers as
            # a leading zero.
            if len(tens_probs) > 10:
                tens_index = 10
            else:
                tens_index = 0
        else:
            tens_index = value // 10

        if tens_index >= len(tens_probs):
            return 0.0
        return float(tens_probs[tens_index] * ones_probs[ones_digit])

    def _decode_roster(
        self,
        team_name: str | None,
        tens_probs: list[float],
        ones_probs: list[float],
    ) -> dict:
        valid = self._valid_roster_numbers(team_name)
        if not valid:
            return {
                "candidate_number": None,
                "candidate_probability": 0.0,
                "candidate_posterior": 0.0,
                "candidate_margin": 0.0,
                "valid_roster_numbers": [],
                "top_candidates": [],
            }

        scored = [
            (number, self._candidate_probability(number, tens_probs, ones_probs))
            for number in valid
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        total = sum(score for _, score in scored) or 1e-12
        best_number, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        posterior = best_score / total
        margin = (best_score - second_score) / total
        return {
            "candidate_number": best_number,
            "candidate_probability": round(best_score, 6),
            "candidate_posterior": round(posterior, 6),
            "candidate_margin": round(max(0.0, margin), 6),
            "valid_roster_numbers": valid,
            "top_candidates": [
                {"number": number, "posterior": round(score / total, 6)}
                for number, score in scored[:3]
            ],
        }

    @staticmethod
    def _raw_prediction(
        tens_probs: list[float],
        ones_probs: list[float],
    ) -> dict:
        tens_pred = int(np.argmax(tens_probs))
        ones_pred = int(np.argmax(ones_probs))

        if ones_pred >= 10:
            raw_number = None
            raw_canonical = None
            raw_reason = "ones_unknown_class"
        elif tens_pred == 10 and len(tens_probs) > 10:
            raw_number = str(ones_pred)
            raw_canonical = raw_number
            raw_reason = "single_digit_blank_tens"
        elif tens_pred >= 10:
            raw_number = None
            raw_canonical = None
            raw_reason = "tens_unknown_class"
        else:
            raw_number = f"{tens_pred}{ones_pred}"
            raw_canonical = str(int(raw_number))
            raw_reason = "digit_pair"

        tens_conf = float(tens_probs[tens_pred])
        ones_conf = float(ones_probs[ones_pred])
        return {
            "raw_number": raw_number,
            "raw_canonical": raw_canonical,
            "raw_decode_reason": raw_reason,
            "raw_probability": round(tens_conf * ones_conf, 6),
            "tens_digit": tens_pred,
            "units_digit": ones_pred,
            "tens_confidence": round(tens_conf, 6),
            "units_confidence": round(ones_conf, 6),
        }

    def predict_jersey(self, track_id: int, team_name: Optional[str] = None) -> dict:
        if track_id < 0 or not self.ready(track_id):
            return {"accepted": False, "rejection_reason": "sequence_not_ready"}

        buffered = self.track_crop_buffers[track_id][-self.sequence_length:]
        tensors = []
        for frame_crop in buffered:
            rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))

        # The checkpoint is a single-frame model. Run the real track crops as one batch,
        # convert every crop to probabilities, then average those probabilities. This
        # preserves temporal evidence without inventing an LSTM that is absent from the
        # saved weights.
        input_tensor = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            tens_logits, ones_logits = self.number_model(input_tensor)
            tens_prob_tensor = torch.softmax(tens_logits, dim=1).mean(dim=0).detach().cpu()
            ones_prob_tensor = torch.softmax(ones_logits, dim=1).mean(dim=0).detach().cpu()

        tens_probs = [float(value) for value in tens_prob_tensor.tolist()]
        ones_probs = [float(value) for value in ones_prob_tensor.tolist()]
        raw = self._raw_prediction(tens_probs, ones_probs)

        decoded = self._decode_roster(team_name, tens_probs, ones_probs)
        candidate = decoded.get("candidate_number")
        posterior = float(decoded.get("candidate_posterior", 0.0) or 0.0)
        margin = float(decoded.get("candidate_margin", 0.0) or 0.0)

        if candidate is None:
            accepted = False
            reason = "no_known_roster_numbers"
        elif posterior < 0.30:
            accepted = False
            reason = "low_roster_posterior"
        elif margin < 0.05:
            accepted = False
            reason = "low_roster_margin"
        else:
            accepted = True
            reason = "accepted"

        cross_team = {}
        for candidate_team in self.rosters:
            team_decode = self._decode_roster(
                str(candidate_team),
                tens_probs,
                ones_probs,
            )
            cross_team[str(candidate_team)] = {
                "candidate_number": team_decode.get("candidate_number"),
                "candidate_posterior": team_decode.get("candidate_posterior", 0.0),
                "candidate_margin": team_decode.get("candidate_margin", 0.0),
            }

        return {
            "accepted": accepted,
            "rejection_reason": reason,
            "team": team_name,
            **raw,
            **decoded,
            "cross_team_best": cross_team,
            "sequence_crops": len(buffered),
            "checkpoint": self.checkpoint_diagnostics,
        }

    def get_shirt_number(
        self,
        crop: np.ndarray,
        track_id: int = 0,
        team_name: Optional[str] = None,
    ) -> str:
        prediction = self.predict_jersey(track_id=track_id, team_name=team_name)
        if not prediction.get("accepted"):
            return ""
        return str(prediction.get("candidate_number") or "")
