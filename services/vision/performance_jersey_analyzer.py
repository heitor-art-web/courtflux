from __future__ import annotations

import json
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
    """EfficientNet-B0 + temporal LSTM architecture for jersey recognition.

    The backbone is intentionally created with weights=None. Performance runs use the
    repository-local checkpoint and must never trigger a torchvision network download.
    """

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
    """Team classification + throttled jersey recognition for detection benchmarks."""

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
        print(
            f"📦 Loading local jersey model from {checkpoint_path} on {self.device.type}...",
            flush=True,
        )

        # Load checkpoint on CPU first. This avoids a large device allocation while the
        # state dict is being deserialized and is more reliable on MPS.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = TemporalJerseyClassifier()

        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        else:
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            if not isinstance(state_dict, dict):
                raise TypeError("Unsupported jersey checkpoint format")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"⚠️ Jersey checkpoint missing {len(missing)} keys", flush=True)
            if unexpected:
                print(f"⚠️ Jersey checkpoint has {len(unexpected)} unexpected keys", flush=True)

        self.number_model = model.to(self.device).eval()
        print(
            f"✅ Jersey model ready in {perf_counter() - started:.2f}s; no external weights requested.",
            flush=True,
        )

    def observe_track(self, crop: np.ndarray, track_id: int) -> None:
        """Collect one real torso crop per frame without running the neural model."""
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
            # Track IDs are insertion ordered. Keep the most recently added entries.
            keep = list(self.track_crop_buffers.keys())[-100:]
            self.track_crop_buffers = {key: self.track_crop_buffers[key] for key in keep}

    def ready(self, track_id: int) -> bool:
        return len(self.track_crop_buffers.get(track_id, [])) >= self.sequence_length

    def get_shirt_number(
        self,
        crop: np.ndarray,
        track_id: int = 0,
        team_name: Optional[str] = None,
    ) -> str:
        """Run one temporal inference only after a real sequence is available."""
        if track_id < 0 or not self.ready(track_id):
            return ""

        buffered = self.track_crop_buffers[track_id][-self.sequence_length :]
        tensors = []
        for frame_crop in buffered:
            rgb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))

        input_tensor = torch.stack(tensors).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            tens_logits, units_logits = self.number_model(input_tensor)
            tens_pred = int(torch.argmax(tens_logits, dim=1).cpu().item())
            units_pred = int(torch.argmax(units_logits, dim=1).cpu().item())

        raw_number = f"{tens_pred}{units_pred}"
        if team_name and team_name in self.rosters:
            roster = (self.rosters.get(team_name, {}) or {}).get("roster", {}) or {}
            valid = {str(key).strip() for key in roster if not str(key).startswith("?")}
            if valid:
                variants = [raw_number]
                if raw_number.isdigit():
                    variants.append(str(int(raw_number)))
                return next((value for value in variants if value in valid), "")

        return raw_number
