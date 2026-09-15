from __future__ import annotations

import math
import os
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


class TemporalJerseyClassifier(nn.Module):
    """Exact runtime reconstruction of the PR #37 best_temporal.pt contract."""

    def __init__(
        self,
        projection_in: int,
        projection_out: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        head_in: int,
        tens_classes: int,
        ones_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.proj = nn.Linear(projection_in, projection_out)
        self.lstm = nn.LSTM(
            input_size=projection_out,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.head_tens = nn.Linear(head_in, tens_classes)
        self.head_ones = nn.Linear(head_in, ones_classes)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.head_in = head_in

    def _temporal_representation(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[1]
        directions = 2 if self.bidirectional else 1
        all_hidden_width = self.num_layers * directions * self.hidden_size
        last_layer_width = directions * self.hidden_size

        if self.head_in == all_hidden_width:
            return hidden.transpose(0, 1).contiguous().reshape(batch, -1)
        if self.head_in == last_layer_width:
            start = (self.num_layers - 1) * directions
            last = hidden[start : start + directions]
            return last.transpose(0, 1).contiguous().reshape(batch, -1)
        if self.head_in == self.hidden_size:
            return hidden[-1]
        raise RuntimeError(
            "Unable to map LSTM hidden states to classifier heads: "
            f"head_in={self.head_in}, all_hidden_width={all_hidden_width}, "
            f"last_layer_width={last_layer_width}."
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,T,C,H,W], got {tuple(x.shape)}")
        batch, steps, channels, height, width = x.shape
        flat = x.reshape(batch * steps, channels, height, width)
        features = self.backbone(flat)
        embeddings = torch.relu(self.proj(features)).reshape(batch, steps, -1)
        _, (hidden, _) = self.lstm(embeddings)
        representation = self._temporal_representation(hidden)
        return self.head_tens(representation), self.head_ones(representation)


def _iter_tensor_dicts(value: Any, path: str = "root", depth: int = 0):
    if depth > 5 or not isinstance(value, dict):
        return
    tensors = {str(key): item for key, item in value.items() if torch.is_tensor(item)}
    if tensors:
        yield path, tensors
    for key, item in value.items():
        if isinstance(item, dict):
            yield from _iter_tensor_dicts(item, f"{path}.{key}", depth + 1)


def _strip_common_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expected = ("backbone.", "proj.", "lstm.", "head_tens.", "head_ones.")
    if any(any(key.startswith(prefix) for prefix in expected) for key in state):
        return state
    for prefix in ("module.", "model.", "network.", "net."):
        stripped = {
            (key[len(prefix) :] if key.startswith(prefix) else key): value
            for key, value in state.items()
        }
        if any(key.startswith("lstm.") for key in stripped):
            return stripped
    return state


def _select_temporal_state(checkpoint: Any) -> tuple[str, dict[str, torch.Tensor]]:
    best: tuple[tuple[int, int, int], str, dict[str, torch.Tensor]] | None = None
    for source, raw in _iter_tensor_dicts(checkpoint):
        state = _strip_common_prefix(raw)
        score = (
            sum(key.startswith("lstm.") for key in state),
            sum(key.startswith("backbone.features.") for key in state),
            len(state),
        )
        if best is None or score > best[0]:
            best = (score, source, state)
    if best is None or best[0][0] == 0:
        raise RuntimeError("Checkpoint contains no temporal LSTM state dictionary")
    return best[1], best[2]


def _infer_architecture(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    required = [
        "proj.weight",
        "proj.bias",
        "lstm.weight_ih_l0",
        "lstm.weight_hh_l0",
        "head_tens.weight",
        "head_tens.bias",
        "head_ones.weight",
        "head_ones.bias",
    ]
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"Temporal checkpoint missing required keys: {missing}")

    projection_out, projection_in = map(int, state["proj.weight"].shape)
    hidden_size = int(state["lstm.weight_hh_l0"].shape[1])
    lstm_input = int(state["lstm.weight_ih_l0"].shape[1])
    if lstm_input != projection_out:
        raise RuntimeError(
            f"LSTM input={lstm_input} does not match projection output={projection_out}"
        )

    layer_indices: list[int] = []
    for key in state:
        match = re.fullmatch(r"lstm\.weight_ih_l(\d+)(?:_reverse)?", key)
        if match:
            layer_indices.append(int(match.group(1)))
    num_layers = max(layer_indices, default=0) + 1
    bidirectional = any(
        key.endswith("_reverse") for key in state if key.startswith("lstm.")
    )
    tens_classes, head_in = map(int, state["head_tens.weight"].shape)
    ones_classes, ones_head_in = map(int, state["head_ones.weight"].shape)
    if ones_head_in != head_in:
        raise RuntimeError("Tens and ones heads use different input widths")
    if tens_classes < 10 or ones_classes < 10:
        raise RuntimeError(
            "best_temporal runtime requires at least ten tens/ones classes; "
            f"got tens={tens_classes} ones={ones_classes}"
        )

    return {
        "projection_in": projection_in,
        "projection_out": projection_out,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "bidirectional": bidirectional,
        "head_in": head_in,
        "tens_classes": tens_classes,
        "ones_classes": ones_classes,
    }


class PerformanceTemporalJerseyAnalyzer(TeamOnlyAnalyzer):
    """Runtime adapter for weights/best_temporal.pt.

    The model itself consumes an ordered eight-frame sequence. Runtime therefore keeps
    a chronological player-crop history and uniformly samples eight observations, which
    matches the PR #37 benchmark contract instead of treating the temporal checkpoint as
    eight independent single-frame predictions.
    """

    BACKEND_NAME = "best_temporal"
    SEQUENCE_LENGTH = 8

    def __init__(
        self,
        profiling_path: str,
        device: str = "mps",
        model_path: str = "weights/best_temporal.pt",
        sequence_length: int = 8,
    ) -> None:
        super().__init__(profiling_path=profiling_path)
        if int(sequence_length) != self.SEQUENCE_LENGTH:
            raise ValueError("best_temporal checkpoint contract requires sequence_length=8")

        self.sequence_length = self.SEQUENCE_LENGTH
        self.history_length = self._env_int(
            "BASKETBALL_JERSEY_TEMPORAL_HISTORY_FRAMES", 48, 8, 240
        )
        self.min_posterior = self._env_float(
            "BASKETBALL_JERSEY_TEMPORAL_POSTERIOR_MIN", 0.18, 0.0, 1.0
        )
        self.min_margin = self._env_float(
            "BASKETBALL_JERSEY_TEMPORAL_MARGIN_MIN", 0.03, 0.0, 1.0
        )
        self.single_digit_tens_min = self._env_float(
            "BASKETBALL_JERSEY_TEMPORAL_SINGLE_TENS_MIN", 0.65, 0.0, 1.0
        )
        self.single_digit_tens_margin_min = self._env_float(
            "BASKETBALL_JERSEY_TEMPORAL_SINGLE_TENS_MARGIN_MIN", 0.25, 0.0, 1.0
        )
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
                f"Temporal jersey checkpoint not found: {checkpoint_path}"
            )

        started = perf_counter()
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        source, state = _select_temporal_state(checkpoint)
        architecture = _infer_architecture(state)
        model = TemporalJerseyClassifier(**architecture)
        model_state = model.state_dict()
        missing = [key for key in model_state if key not in state]
        unexpected = [key for key in state if key not in model_state]
        mismatches = [
            (key, tuple(state[key].shape), tuple(model_state[key].shape))
            for key in model_state
            if key in state and tuple(state[key].shape) != tuple(model_state[key].shape)
        ]
        if missing or unexpected or mismatches:
            raise RuntimeError(
                "Temporal checkpoint does not exactly match reconstructed architecture: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"mismatches={mismatches[:5]}"
            )
        model.load_state_dict(state, strict=True)
        self.number_model = model.to(self.device).eval()
        self.model_path = checkpoint_path
        self.transform = transforms.Compose(
            [
                PadToSquare(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        parameter_count = int(sum(p.numel() for p in model.parameters()))
        self.checkpoint_diagnostics = {
            "backend": self.BACKEND_NAME,
            "source": source,
            "architecture": "efficientnet_b0_projection_bilstm_digit_heads_v1",
            "checkpoint": str(checkpoint_path),
            "parameter_count": parameter_count,
            "img_size": 128,
            "sequence_length": self.sequence_length,
            "projection_in": architecture["projection_in"],
            "projection_out": architecture["projection_out"],
            "hidden_size": architecture["hidden_size"],
            "num_layers": architecture["num_layers"],
            "bidirectional": architecture["bidirectional"],
            "head_in": architecture["head_in"],
            "tens_classes": architecture["tens_classes"],
            "ones_classes": architecture["ones_classes"],
            "preprocessing": "player_crop_pad_square_resize128_rgb_imagenet",
            "temporal_aggregation": "ordered_uniform_8_frame_sequence_bilstm",
            "roster_policy": "validator_only",
            "acceptance_gate": {
                "candidate_posterior_min": self.min_posterior,
                "candidate_margin_min": self.min_margin,
                "single_digit_tens_probability_min": self.single_digit_tens_min,
                "single_digit_tens_margin_min": self.single_digit_tens_margin_min,
            },
            "historical_source": "PR37 tools/jersey_temporal_ocr_benchmark.py",
        }
        print(
            "🪪 Jersey backend: best_temporal "
            f"checkpoint={checkpoint_path.name} device={self.device.type} "
            f"sequence=8 params={parameter_count:,} "
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
        return max(minimum, min(maximum, value))

    @staticmethod
    def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
        raw = os.getenv(name)
        try:
            value = float(raw) if raw is not None and raw.strip() else float(default)
        except (TypeError, ValueError):
            value = float(default)
        return max(minimum, min(maximum, value))

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
        resolution = min(area / float(120 * 60), 1.0)
        return float(
            0.36 * sharpness
            + 0.24 * contrast
            + 0.24 * exposure
            + 0.16 * resolution
        )

    @staticmethod
    def _uniform_indices(count: int, target: int = 8) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        if target <= 0:
            raise ValueError("target must be positive")
        if count == 1:
            return [0] * target
        return np.rint(np.linspace(0, count - 1, target)).astype(int).tolist()

    @staticmethod
    def _top_digits(probabilities: np.ndarray, limit: int = 3) -> list[dict[str, Any]]:
        order = np.argsort(probabilities)[::-1][: max(1, int(limit))]
        return [
            {"digit": int(index), "probability": round(float(probabilities[index]), 6)}
            for index in order
        ]

    @staticmethod
    def _joint_number_distribution(
        tens_probs: np.ndarray, ones_probs: np.ndarray
    ) -> np.ndarray:
        scores = np.zeros(100, dtype=np.float64)
        for number in range(100):
            scores[number] = float(
                tens_probs[number // 10] * ones_probs[number % 10]
            )
        return scores

    @staticmethod
    def _top_numbers(probabilities: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
        order = np.argsort(probabilities)[::-1][: max(1, int(limit))]
        return [
            {"number": str(int(index)), "posterior": round(float(probabilities[index]), 6)}
            for index in order
        ]

    def _valid_roster_numbers(self, team_name: str | None) -> list[str]:
        if not team_name or team_name not in self.rosters:
            return []
        roster = (self.rosters.get(team_name, {}) or {}).get("roster", {}) or {}
        values: list[str] = []
        for key in roster:
            text = str(key).strip()
            if text.isdigit() and 0 <= int(text) <= 99:
                values.append(str(int(text)))
        return sorted(set(values), key=int)

    def observe_track(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_id: int | None = None,
    ) -> None:
        if track_id < 0 or crop is None or crop.size == 0:
            return
        height, width = crop.shape[:2]
        if height < 40 or width < 20:
            return
        ordinal = int(self.track_observation_index.get(track_id, 0)) + 1
        self.track_observation_index[track_id] = ordinal
        buffer = list(self.track_crop_buffers.get(track_id, []))
        buffer.append(
            {
                "crop": crop.copy(),
                "quality": self._crop_quality(crop),
                "ordinal": ordinal,
                "frame_id": frame_id,
            }
        )
        self.track_crop_buffers[track_id] = buffer[-self.history_length :]
        if len(self.track_crop_buffers) > 250:
            keep = sorted(
                self.track_crop_buffers,
                key=lambda key: int(self.track_observation_index.get(key, 0)),
                reverse=True,
            )[:100]
            self.track_crop_buffers = {
                key: self.track_crop_buffers[key] for key in keep
            }
            self.track_observation_index = {
                key: self.track_observation_index.get(key, 0) for key in keep
            }

    def ready(self, track_id: int) -> bool:
        return len(self.track_crop_buffers.get(track_id, [])) >= self.sequence_length

    def predict_jersey(
        self, track_id: int, team_name: str | None = None
    ) -> dict[str, Any]:
        history = list(self.track_crop_buffers.get(track_id, []))
        if track_id < 0 or len(history) < self.sequence_length:
            return {
                "accepted": False,
                "rejection_reason": "sequence_not_ready",
                "roster_policy": "validator_only",
            }

        history.sort(key=lambda item: int(item.get("ordinal", 0)))
        selected = [history[index] for index in self._uniform_indices(len(history), 8)]
        tensors: list[torch.Tensor] = []
        for item in selected:
            rgb = cv2.cvtColor(item["crop"], cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        sequence = torch.stack(tensors).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            tens_logits, ones_logits = self.number_model(sequence)
            tens_probs = torch.softmax(tens_logits, dim=1)[0].detach().float().cpu().numpy()
            ones_probs = torch.softmax(ones_logits, dim=1)[0].detach().float().cpu().numpy()

        joint = self._joint_number_distribution(tens_probs, ones_probs)
        order = np.argsort(joint)[::-1]
        winner = int(order[0])
        candidate = str(winner)
        posterior = float(joint[order[0]])
        second = float(joint[order[1]]) if len(order) > 1 else 0.0
        margin = max(0.0, posterior - second)
        tens_digit, ones_digit = winner // 10, winner % 10
        tens_order = np.argsort(tens_probs)[::-1]
        tens_second = float(tens_probs[tens_order[1]]) if len(tens_order) > 1 else 0.0
        single_required = winner <= 9
        single_passed = bool(
            not single_required
            or (
                tens_digit == 0
                and float(tens_probs[0]) >= self.single_digit_tens_min
                and float(tens_probs[0]) - tens_second >= self.single_digit_tens_margin_min
            )
        )
        valid = self._valid_roster_numbers(team_name)
        roster_valid = candidate in valid

        if single_required and not single_passed:
            accepted, reason = False, "ambiguous_single_digit_tens"
        elif posterior < self.min_posterior:
            accepted, reason = False, "low_raw_consensus_posterior"
        elif margin < self.min_margin:
            accepted, reason = False, "low_raw_consensus_margin"
        elif not team_name:
            accepted, reason = False, "team_unknown_for_roster_validation"
        elif not valid:
            accepted, reason = False, "no_known_roster_numbers"
        elif not roster_valid:
            accepted, reason = False, "raw_not_in_roster"
        else:
            accepted, reason = True, "accepted"

        # Joint tens×ones probability is useful diagnostically but is too compressed to
        # be reused directly as an Identity-v2 evidence weight. The geometric mean
        # recovers the per-head confidence scale while preserving conservative ordering.
        evidence = math.sqrt(max(0.0, posterior))
        cross_team_validation: dict[str, Any] = {}
        for candidate_team in self.rosters:
            team_valid = self._valid_roster_numbers(str(candidate_team))
            is_valid = candidate in team_valid
            cross_team_validation[str(candidate_team)] = {
                "candidate_number": candidate if is_valid else None,
                "candidate_posterior": round(posterior, 6) if is_valid else 0.0,
                "candidate_margin": round(margin, 6) if is_valid else 0.0,
                "roster_valid": is_valid,
            }

        return {
            "accepted": accepted,
            "rejection_reason": reason,
            "team": team_name,
            "candidate_number": candidate,
            "raw_number": candidate,
            "raw_canonical": candidate,
            "raw_decode_reason": "best_temporal_bilstm_sequence",
            "candidate_probability": round(posterior, 6),
            "candidate_posterior": round(posterior, 6),
            "candidate_margin": round(margin, 6),
            "candidate_count": len(selected),
            "candidate_evidence": round(evidence, 6),
            "candidate_evidence_source": "geometric_mean_digit_probability",
            "top_candidates": self._top_numbers(joint),
            "tens_top3": self._top_digits(tens_probs),
            "ones_top3": self._top_digits(ones_probs),
            "single_digit_gate": {
                "required": single_required,
                "passed": single_passed,
                "reason": (
                    "not_applicable_two_digit"
                    if not single_required
                    else "temporal_no_tens_evidence"
                    if single_passed
                    else "ambiguous_single_digit_tens"
                ),
                "tens_zero_probability": round(float(tens_probs[0]), 6),
                "tens_zero_margin": round(float(tens_probs[0]) - tens_second, 6),
                "probability_min": self.single_digit_tens_min,
                "margin_min": self.single_digit_tens_margin_min,
            },
            "valid_roster_numbers": valid,
            "roster_valid": roster_valid,
            "roster_policy": "validator_only",
            "cross_team_validation": cross_team_validation,
            "cross_team_best": cross_team_validation,
            "sequence_crops": len(selected),
            "sequence_history_crops": len(history),
            "buffer_policy": "chronological_history_uniform_8",
            "aggregation": "bilstm_sequence",
            "crop_quality": [round(float(item.get("quality", 0.0)), 6) for item in selected],
            "per_crop_predictions": [
                {
                    "frame_id": item.get("frame_id"),
                    "crop_quality": round(float(item.get("quality", 0.0)), 6),
                    "sequence_member": True,
                }
                for item in selected
            ],
            "checkpoint": self.checkpoint_diagnostics,
            "p1_raw_roster_consistency_gate": True,
        }
