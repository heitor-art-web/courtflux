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


class TemporalJerseyClassifier(nn.Module):
    """Checkpoint-native temporal jersey recognizer.

    The local checkpoint uses this exact module layout:
      backbone.features.*
      feature_projection.0.*
      lstm.*
      tens_head.{0,3}.*
      units_head.{0,3}.*

    Parameter count is 5,194,384, matching the published temporal EfficientNet-B0 +
    2-layer BiLSTM architecture. No external pretrained weights are requested at
    inference time; every learned parameter comes from the repository-local checkpoint.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()

        self.feature_projection = nn.Sequential(
            nn.Linear(1280, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_output_dim = lstm_hidden * 2
        self.tens_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )
        self.units_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor):
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(batch_size * sequence_length, channels, height, width)
        features = self.backbone(x)
        features = self.feature_projection(features)
        features = features.view(batch_size, sequence_length, -1)
        lstm_out, _ = self.lstm(features)
        final_features = lstm_out[:, -1, :]
        return self.tens_head(final_features), self.units_head(final_features)


class PerformanceJerseyAnalyzer(TeamOnlyAnalyzer):
    """Team scoring + structured temporal jersey decoding for beta identity runs."""

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

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Jersey checkpoint not found: {checkpoint_path}")

        started = perf_counter()
        print(f"📦 Loading local jersey model from {checkpoint_path} on {self.device.type}...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model_state = model.state_dict()
            state_source = "root.nn_module"
            shape_matches = len(model_state)
            key_matches = len(model_state)
            tensor_keys = len(model_state)
            coverage = 1.0
            missing_count = 0
            unexpected_count = 0
        else:
            model = TemporalJerseyClassifier()
            model_state = model.state_dict()
            state_source, state_dict, inspection = self._select_state_dict(checkpoint, model_state)
            shape_matches = int(inspection["shape_matches"])
            key_matches = int(inspection["key_matches"])
            tensor_keys = int(inspection["tensor_keys"])
            coverage = shape_matches / max(len(model_state), 1)

            print(
                "🔎 Jersey checkpoint inspection: "
                f"source={state_source} "
                f"shape_matches={shape_matches}/{len(model_state)} "
                f"key_matches={key_matches} tensor_keys={tensor_keys}",
                flush=True,
            )

            if coverage < 0.98:
                sample_keys = ", ".join(list(state_dict.keys())[:8])
                tail_keys = ", ".join(list(state_dict.keys())[-8:])
                raise RuntimeError(
                    "Jersey checkpoint architecture mismatch after nested checkpoint inspection. "
                    f"Best tensor dictionary='{state_source}', coverage={coverage:.1%} "
                    f"({shape_matches}/{len(model_state)} shape-compatible model keys). "
                    f"Sample checkpoint keys: {sample_keys}. Tail checkpoint keys: {tail_keys}"
                )

            compatible = {
                key: value
                for key, value in state_dict.items()
                if key in model_state
                and torch.is_tensor(value)
                and tuple(value.shape) == tuple(model_state[key].shape)
            }
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            missing_count = len(missing)
            unexpected_count = len(unexpected)

            if missing_count:
                raise RuntimeError(
                    f"Jersey checkpoint passed compatibility scoring but still left "
                    f"{missing_count} model keys unloaded. First missing keys: {list(missing)[:8]}"
                )

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.checkpoint_diagnostics = {
            "source": state_source,
            "coverage": round(float(coverage), 4),
            "shape_matches": int(shape_matches),
            "key_matches": int(key_matches),
            "tensor_keys": int(tensor_keys),
            "model_state_keys": int(len(model.state_dict())),
            "missing_keys": int(missing_count),
            "unexpected_keys": int(unexpected_count),
            "parameter_count": int(parameter_count),
            "architecture": "efficientnet_b0+projection256+bilstm128x2+digit_heads",
        }

        self.number_model = model.to(self.device).eval()
        print(
            f"✅ Jersey model ready in {perf_counter() - started:.2f}s; "
            f"checkpoint coverage={coverage:.1%}; params={parameter_count:,}.",
            flush=True,
        )

    @staticmethod
    def _iter_tensor_dicts(value, path: str = "root", depth: int = 0) -> Iterable[tuple[str, dict]]:
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
    def _select_state_dict(cls, checkpoint, model_state: dict) -> tuple[str, dict, dict]:
        candidates = list(cls._iter_tensor_dicts(checkpoint)) if isinstance(checkpoint, dict) else []
        if not candidates:
            raise TypeError(
                "Unsupported jersey checkpoint format: no tensor state dictionary found."
            )

        best = None
        for source, raw_state in candidates:
            for variant_name, candidate in cls._state_variants(raw_state):
                key_matches = sum(1 for key in candidate if key in model_state)
                shape_matches = sum(
                    1
                    for key, value in candidate.items()
                    if key in model_state
                    and torch.is_tensor(value)
                    and tuple(value.shape) == tuple(model_state[key].shape)
                )
                score = (shape_matches, key_matches, -abs(len(candidate) - len(model_state)))
                if best is None or score > best[0]:
                    best = (
                        score,
                        f"{source}[{variant_name}]",
                        candidate,
                        {
                            "shape_matches": shape_matches,
                            "key_matches": key_matches,
                            "tensor_keys": len(candidate),
                        },
                    )

        assert best is not None
        return best[1], best[2], best[3]

    def observe_track(self, crop: np.ndarray, track_id: int) -> None:
        """Collect one genuine torso observation per frame for the temporal sequence."""
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
        units_probs: list[float],
    ) -> float:
        value = int(number)
        return float(tens_probs[value // 10] * units_probs[value % 10])

    def _decode_roster(
        self,
        team_name: str | None,
        tens_probs: list[float],
        units_probs: list[float],
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
            (number, self._candidate_probability(number, tens_probs, units_probs))
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

    def predict_jersey(self, track_id: int, team_name: Optional[str] = None) -> dict:
        if track_id < 0 or not self.ready(track_id):
            return {"accepted": False, "rejection_reason": "sequence_not_ready"}

        buffered = self.track_crop_buffers[track_id][-self.sequence_length:]
        tensors = []
        for frame_crop in buffered:
            rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))

        input_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            tens_logits, units_logits = self.number_model(input_tensor)
            tens_prob_tensor = torch.softmax(tens_logits, dim=1)[0].detach().cpu()
            units_prob_tensor = torch.softmax(units_logits, dim=1)[0].detach().cpu()

        tens_probs = [float(value) for value in tens_prob_tensor.tolist()]
        units_probs = [float(value) for value in units_prob_tensor.tolist()]
        tens_pred = int(np.argmax(tens_probs))
        units_pred = int(np.argmax(units_probs))
        raw_number = f"{tens_pred}{units_pred}"
        raw_canonical = str(int(raw_number))
        raw_probability = float(tens_probs[tens_pred] * units_probs[units_pred])

        decoded = self._decode_roster(team_name, tens_probs, units_probs)
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
                units_probs,
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
            "raw_number": raw_number,
            "raw_canonical": raw_canonical,
            "raw_probability": round(raw_probability, 6),
            "tens_digit": tens_pred,
            "units_digit": units_pred,
            "tens_confidence": round(tens_probs[tens_pred], 6),
            "units_confidence": round(units_probs[units_pred], 6),
            **decoded,
            "cross_team_best": cross_team,
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
