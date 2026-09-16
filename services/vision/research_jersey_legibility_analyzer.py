from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from services.vision.research_jersey_region_analyzer import (
    ResearchNumberRegionPARSeqAnalyzer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOCCERNET_LEGIBILITY_GDRIVE_ID = "18HAuZbge3z8TSfRiX_FzsnKgiBs-RRNw"
DEFAULT_LEGIBILITY_CHECKPOINT = (
    PROJECT_ROOT / "weights" / "legibility_resnet34_soccer_20240215.pth"
)


class _SoccerNetLegibilityResNet34(nn.Module):
    """Architecture used by Koshkina/Elder for SoccerNet legibility inference."""

    def __init__(self) -> None:
        super().__init__()
        self.model_ft = models.resnet34(weights=None)
        num_features = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model_ft(inputs))


class ResearchLegibilityRegionPARSeqAnalyzer(ResearchNumberRegionPARSeqAnalyzer):
    """Research v5: SoccerNet legibility gate before v4 number-region + PARSeq.

    Legibility is evaluated on the original full player crop, matching the SoccerNet
    classifier's intended input domain. The downstream v4 region scorer and PARSeq still
    operate on CourtFlux's focused torso crop. The gate is roster-blind and OCR-blind.
    """

    LEGIBILITY_THRESHOLD_DEFAULT = 0.50

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        super().__init__(profiling_path=profiling_path, device=device)
        self.legibility_threshold = self._env_float(
            "BASKETBALL_JERSEY_RESEARCH_LEGIBILITY_THRESHOLD",
            self.LEGIBILITY_THRESHOLD_DEFAULT,
            0.05,
            0.95,
        )
        configured = str(
            os.getenv("BASKETBALL_JERSEY_LEGIBILITY_CHECKPOINT", "") or ""
        ).strip()
        checkpoint_path = (
            Path(configured).expanduser()
            if configured
            else DEFAULT_LEGIBILITY_CHECKPOINT
        )
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        self.legibility_checkpoint = checkpoint_path.resolve()
        self._ensure_legibility_checkpoint()

        payload = torch.load(
            str(self.legibility_checkpoint), map_location="cpu", weights_only=False
        )
        state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
        if not isinstance(state_dict, dict):
            state_dict = payload
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(
                "Invalid SoccerNet legibility checkpoint: expected a non-empty state_dict"
            )
        normalized_state = {
            (str(key)[7:] if str(key).startswith("module.") else str(key)): value
            for key, value in state_dict.items()
        }
        self.legibility_model = _SoccerNetLegibilityResNet34()
        try:
            self.legibility_model.load_state_dict(normalized_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "SoccerNet legibility checkpoint is incompatible with ResNet34 architecture: "
                f"{exc}"
            ) from exc
        self.legibility_model = self.legibility_model.to(self.device).eval()
        self._legibility_cache: dict[tuple[int, int], float] = {}
        self._last_legibility_gate: dict[int, dict[str, Any]] = {}

        self.checkpoint_diagnostics.update(
            {
                "temporal_aggregation": "research_legibility_region_then_identity_v2",
                "legibility_gate": True,
                "legibility_model": "Koshkina-Elder SoccerNet ResNet34",
                "legibility_checkpoint": str(self.legibility_checkpoint),
                "legibility_checkpoint_source": "official_cvprw2024_google_drive",
                "legibility_checkpoint_gdrive_id": SOCCERNET_LEGIBILITY_GDRIVE_ID,
                "legibility_threshold": self.legibility_threshold,
                "legibility_input": "full_player_crop_256x256_imagenet_norm",
                "pose_guided_roi": False,
            }
        )
        print(
            "[JERSEY_LEGIBILITY] "
            f"model=resnet34_soccernet threshold={self.legibility_threshold:.2f} "
            f"checkpoint={self.legibility_checkpoint}",
            flush=True,
        )

    def _ensure_legibility_checkpoint(self) -> None:
        if self.legibility_checkpoint.exists() and self.legibility_checkpoint.stat().st_size > 0:
            return
        if not self._env_bool("BASKETBALL_JERSEY_LEGIBILITY_AUTO_DOWNLOAD", True):
            raise FileNotFoundError(
                f"SoccerNet legibility checkpoint is missing: {self.legibility_checkpoint}"
            )
        self.legibility_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = self.legibility_checkpoint.with_suffix(
            self.legibility_checkpoint.suffix + ".part"
        )
        if partial.exists():
            partial.unlink()
        downloaded = gdown.download(
            id=SOCCERNET_LEGIBILITY_GDRIVE_ID,
            output=str(partial),
            quiet=False,
        )
        if not downloaded or not partial.exists() or partial.stat().st_size <= 0:
            if partial.exists():
                partial.unlink()
            raise RuntimeError("Could not download official SoccerNet legibility checkpoint")
        partial.replace(self.legibility_checkpoint)

    def observe_track(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_id: int | None = None,
    ) -> None:
        """Keep the full player crop for legibility while preserving v4 torso evidence."""
        track_id = int(track_id)
        previous_ordinal = int(self.track_observation_index.get(track_id, 0))
        super().observe_track(crop, track_id, frame_id=frame_id)
        current_ordinal = int(self.track_observation_index.get(track_id, 0))
        if current_ordinal <= previous_ordinal:
            return
        buffer = self.track_crop_buffers.get(track_id)
        if not buffer:
            return
        newest = buffer[-1]
        if int(newest.get("ordinal", -1)) == current_ordinal:
            newest["legibility_crop"] = crop.copy()

    @staticmethod
    def _prepare_legibility_crop(crop: np.ndarray) -> torch.Tensor:
        if crop is None or crop.size == 0:
            raise ValueError("legibility crop is empty")
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def _score_legibility_items(
        self, track_id: int, items: list[dict[str, Any]]
    ) -> list[float]:
        scores: list[float | None] = [None] * len(items)
        missing_indices: list[int] = []
        missing_tensors: list[torch.Tensor] = []
        for index, item in enumerate(items):
            ordinal = int(item.get("ordinal", 0))
            cache_key = (int(track_id), ordinal)
            cached = self._legibility_cache.get(cache_key)
            if cached is not None:
                scores[index] = float(cached)
                continue
            source = item.get("legibility_crop")
            if source is None or not isinstance(source, np.ndarray) or source.size == 0:
                source = item.get("crop")
            if source is None or not isinstance(source, np.ndarray) or source.size == 0:
                scores[index] = 0.0
                self._legibility_cache[cache_key] = 0.0
                continue
            missing_indices.append(index)
            missing_tensors.append(self._prepare_legibility_crop(source))

        if missing_tensors:
            batch = torch.stack(missing_tensors, dim=0).to(self.device)
            with torch.inference_mode():
                outputs = self.legibility_model(batch).reshape(-1)
            values = outputs.detach().float().cpu().tolist()
            for index, value in zip(missing_indices, values):
                score = max(0.0, min(1.0, float(value)))
                scores[index] = score
                ordinal = int(items[index].get("ordinal", 0))
                self._legibility_cache[(int(track_id), ordinal)] = score

        if len(self._legibility_cache) > 5000:
            valid_keys = {
                (int(tid), int(item.get("ordinal", 0)))
                for tid, history in self.track_crop_buffers.items()
                for item in history
            }
            self._legibility_cache = {
                key: value
                for key, value in self._legibility_cache.items()
                if key in valid_keys
            }
        return [float(value or 0.0) for value in scores]

    @staticmethod
    def _gate_ranked_items(
        items: list[dict[str, Any]],
        scores: list[float],
        *,
        threshold: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        passed: list[dict[str, Any]] = []
        for item, score in zip(items, scores):
            if float(score) < float(threshold):
                continue
            copy = dict(item)
            copy["legibility_model_score"] = float(score)
            passed.append(copy)
        selected = sorted(
            passed,
            key=lambda item: (
                float(item.get("legibility_model_score", 0.0)),
                float(item.get("quality", 0.0)),
                int(item.get("ordinal", 0)),
            ),
            reverse=True,
        )[: max(1, int(top_k))]
        selected.sort(key=lambda item: int(item.get("ordinal", 0)))
        return selected

    def _research_select_observations(self, track_id: int) -> list[dict[str, Any]]:
        history = list(self.track_crop_buffers.get(int(track_id), []))
        if not history:
            self._last_legibility_gate[int(track_id)] = {
                "threshold": self.legibility_threshold,
                "scored_count": 0,
                "passed_count": 0,
                "rejected_count": 0,
                "scores": [],
            }
            return []

        last_used = int(self.track_last_inference_ordinal.get(int(track_id), 0))
        fresh = [item for item in history if int(item.get("ordinal", 0)) > last_used]
        pool = fresh if len(fresh) >= self.research_min_support else history
        scores = self._score_legibility_items(int(track_id), pool)
        selected = self._gate_ranked_items(
            pool,
            scores,
            threshold=self.legibility_threshold,
            top_k=self.research_top_k,
        )
        if pool:
            newest_evaluated = max(int(item.get("ordinal", 0)) for item in pool)
            self.track_last_inference_ordinal[int(track_id)] = max(
                last_used, newest_evaluated
            )

        score_rows = [
            {
                "frame_id": item.get("frame_id"),
                "ordinal": int(item.get("ordinal", 0)),
                "score": round(float(score), 6),
                "passed": bool(float(score) >= self.legibility_threshold),
            }
            for item, score in zip(pool, scores)
        ]
        self._last_legibility_gate[int(track_id)] = {
            "threshold": round(self.legibility_threshold, 6),
            "scored_count": len(pool),
            "passed_count": sum(1 for score in scores if score >= self.legibility_threshold),
            "rejected_count": sum(1 for score in scores if score < self.legibility_threshold),
            "mean_score": round(float(np.mean(scores)) if scores else 0.0, 6),
            "max_score": round(max(scores) if scores else 0.0, 6),
            "scores": score_rows,
            "selected_frame_ids": [item.get("frame_id") for item in selected],
        }
        return selected

    def predict_jersey(
        self, track_id: int, team_name: str | None = None
    ) -> dict[str, Any]:
        prediction = super().predict_jersey(track_id=int(track_id), team_name=team_name)
        gate = dict(self._last_legibility_gate.get(int(track_id), {}) or {})
        prediction["legibility_gate"] = gate
        prediction["mean_legibility_score"] = gate.get("mean_score", 0.0)
        if not gate.get("passed_count") and prediction.get("rejection_reason") == "crop_not_ready":
            prediction["rejection_reason"] = "legibility_rejected"
            prediction.setdefault("track_id", int(track_id))
            prediction.setdefault("team", team_name)
            prediction.setdefault("candidate_count", 0)
            prediction.setdefault("per_crop_predictions", [])
            prediction.setdefault(
                "multiframe_consensus",
                {
                    "support": 0,
                    "share": 0.0,
                    "margin": 0.0,
                    "min_support": self.research_min_support,
                    "min_share": self.research_min_share,
                    "selected_images": 0,
                    "roi_enabled": True,
                },
            )
        return prediction
