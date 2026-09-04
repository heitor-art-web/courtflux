from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


class TemporalJerseyClassifier(nn.Module):
    """EfficientNet-B0 + temporal LSTM architecture for jersey recognition."""

    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Linear(1280, 256)
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.tens_head = nn.Linear(512, 10)
        self.units_head = nn.Linear(512, 10)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)
        feat = self.projector(feat)
        feat = feat.view(b, t, -1)
        lstm_out, _ = self.lstm(feat)
        rep = torch.cat([lstm_out[:, -1, :128], lstm_out[:, 0, 128:]], dim=1)
        return self.tens_head(rep), self.units_head(rep)


class PerformanceJerseyAnalyzer(TeamOnlyAnalyzer):
    """Team scoring + structured temporal jersey decoding for beta identity runs.

    The model always exposes its raw digit prediction. Roster constraints are applied as
    a decoder over all valid roster numbers rather than as a binary post-hoc rejection
    of the top raw number. This makes failures diagnosable and uses roster information
    in the statistically useful place: candidate selection.
    """

    def __init__(
        self,
        profiling_path: str,
        device: str = "mps",
        model_path: str = "weights/jersey_number_recon.pt",
        sequence_length: int = 8,
    ):
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
            coverage = 1.0
            missing_count = 0
            unexpected_count = 0
        else:
            model = TemporalJerseyClassifier()
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state_dict, dict):
                raise TypeError("Unsupported jersey checkpoint format")
            state_dict = self._normalize_state_dict(state_dict, set(model.state_dict().keys()))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            model_key_count = max(len(model.state_dict()), 1)
            missing_count = len(missing)
            unexpected_count = len(unexpected)
            coverage = 1.0 - (missing_count / model_key_count)
            if coverage < 0.95:
                raise RuntimeError(
                    f"Jersey checkpoint only covers {coverage:.1%} of model keys "
                    f"({missing_count} missing, {unexpected_count} unexpected)."
                )
            if missing_count or unexpected_count:
                print(
                    f"⚠️ Jersey checkpoint compatibility: coverage={coverage:.1%}, "
                    f"missing={missing_count}, unexpected={unexpected_count}",
                    flush=True,
                )

        self.checkpoint_diagnostics = {
            "coverage": round(float(coverage), 4),
            "missing_keys": int(missing_count),
            "unexpected_keys": int(unexpected_count),
        }
        self.number_model = model.to(self.device).eval()
        print(
            f"✅ Jersey model ready in {perf_counter() - started:.2f}s; "
            f"checkpoint coverage={coverage:.1%}.",
            flush=True,
        )

    @staticmethod
    def _normalize_state_dict(state_dict: dict, model_keys: set[str]) -> dict:
        if set(state_dict.keys()) & model_keys:
            return state_dict
        for prefix in ("module.", "model.", "number_model."):
            candidate = {
                (key[len(prefix):] if str(key).startswith(prefix) else str(key)): value
                for key, value in state_dict.items()
            }
            if set(candidate.keys()) & model_keys:
                return candidate
        return state_dict

    def observe_track(self, crop: np.ndarray, track_id: int) -> None:
        if track_id < 0 or crop is None or crop.size == 0:
            return
        h, w = crop.shape[:2]
        if h < 40 or w < 25:
            return

        ymin, ymax = int(h * 0.15), int(h * 0.65)
        xmin, xmax = int(w * 0.10), int(w * 0.90)
        torso = crop[ymin:ymax, xmin:xmax]
        if torso.size == 0:
            return

        buf = self.track_crop_buffers.setdefault(track_id, [])
        buf.append(torso.copy())
        if len(buf) > self.sequence_length:
            del buf[:-self.sequence_length]

        if len(self.track_crop_buffers) > 250:
            keep = list(self.track_crop_buffers.keys())[-100:]
            self.track_crop_buffers = {key: self.track_crop_buffers[key] for key in keep}

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
    def _candidate_probability(number: str, tens_probs: list[float], units_probs: list[float]) -> float:
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

        scored = [(number, self._candidate_probability(number, tens_probs, units_probs)) for number in valid]
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

        buffered = self.track_crop_buffers[track_id][-self.sequence_length :]
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
            team_decode = self._decode_roster(str(candidate_team), tens_probs, units_probs)
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
        return str(prediction.get("candidate_number") or "") if prediction.get("accepted") else ""
