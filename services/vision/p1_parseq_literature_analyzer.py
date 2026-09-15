from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import models

from services.vision.p1_parseq_jersey_analyzer import (
    PROJECT_ROOT,
    SoccerNetPARSeqJerseyAnalyzer,
)


HOCKEY_LEGIBILITY_GDRIVE_ID = "1RfxINtZ_wCNVF8iZsiMYuFOP7KMgqgDp"
DEFAULT_LEGIBILITY_CHECKPOINT = PROJECT_ROOT / "weights" / "legibility_hockey_resnet34.pth"


class LiteraturePARSeqJerseyAnalyzer(SoccerNetPARSeqJerseyAnalyzer):
    """Literature-guided PARSeq challenger using the same SoccerNet reader.

    This keeps the exact SoccerNet-finetuned PARSeq checkpoint from the baseline
    challenger, but changes the support pipeline around it based on the jersey-number
    literature:

    * player-level legibility filtering using the Hockey-trained ResNet34 released by
      Koshkina & Elder (the same legibility choice used by TrackID3x3 Outdoor);
    * several deterministic torso/number ROI proposals that approximate the
      shoulders-to-hips localization step without adding a pose model yet;
    * one visual vote per frame, followed by confidence-weighted multi-frame consensus;
    * roster validation only after the raw visual consensus is formed.

    The point of this backend is an isolated A/B test: same PARSeq recognizer, better
    observation selection/localization/aggregation. If this helps materially, a true
    pose-guided ROI can be evaluated as a later refinement.
    """

    BACKEND_NAME = "parseq_soccernet_lit"
    LEGIBILITY_SIZE = 256
    ROI_PROPOSALS = (
        # name, y1, y2, x1, x2 as fractions of the full player crop.
        ("torso_wide", 0.08, 0.72, 0.06, 0.94),
        ("number_center", 0.18, 0.60, 0.14, 0.86),
        ("number_upper", 0.12, 0.52, 0.10, 0.90),
    )

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        # Load the exact same SoccerNet PARSeq architecture/checkpoint as the baseline.
        super().__init__(profiling_path=profiling_path, device=device)

        # The base analyzer normally stores an already-focused torso crop. For the
        # literature path we need the full player crop first: the legibility classifier
        # was trained on player images, and ROI localization happens afterwards.
        self.crop_mode = "player"
        self.crop_buffer_size = max(self.crop_buffer_size, 24)

        self.max_frames = self._env_int(
            "BASKETBALL_JERSEY_PARSEQ_LIT_MAX_FRAMES", 4, 2, 8
        )
        self.legibility_threshold = self._env_float(
            "BASKETBALL_JERSEY_PARSEQ_LEGIBILITY_MIN", 0.50, 0.0, 1.0
        )
        self.min_frame_support = self._env_int(
            "BASKETBALL_JERSEY_PARSEQ_MIN_FRAME_SUPPORT", 2, 1, 6
        )
        self.min_vote_share = self._env_float(
            "BASKETBALL_JERSEY_PARSEQ_MIN_VOTE_SHARE", 0.60, 0.0, 1.0
        )
        self.proposal_margin = self._env_float(
            "BASKETBALL_JERSEY_PARSEQ_PROPOSAL_MARGIN", 0.05, 0.0, 1.0
        )

        configured_legibility = str(
            os.getenv("BASKETBALL_JERSEY_LEGIBILITY_CHECKPOINT", "") or ""
        ).strip()
        legibility_path = (
            Path(configured_legibility).expanduser()
            if configured_legibility
            else DEFAULT_LEGIBILITY_CHECKPOINT
        )
        if not legibility_path.is_absolute():
            legibility_path = PROJECT_ROOT / legibility_path
        self.legibility_checkpoint = legibility_path.resolve()
        self._ensure_legibility_checkpoint()
        self.legibility_model = self._load_legibility_model()

        self.checkpoint_diagnostics.update(
            {
                "backend": self.BACKEND_NAME,
                "architecture": "parseq_soccernet_finetuned_plus_legibility_roi_consensus",
                "model_name": "Koshkina-Elder SoccerNet PARSeq (same reader)",
                "preprocessing": (
                    "full_player_legibility_256_imagenet_then_3_pose_free_roi_proposals_"
                    "resize_32x128_rgb_minus1_plus1"
                ),
                "temporal_aggregation": "confidence_weighted_one_vote_per_frame_then_identity_v2",
                "max_images_per_attempt": self.max_frames,
                "legibility_gate": True,
                "legibility_model": "Koshkina-Elder Hockey ResNet34",
                "legibility_checkpoint": str(self.legibility_checkpoint),
                "legibility_threshold": self.legibility_threshold,
                "roi_strategy": "pose_free_shoulders_to_hips_proposal_ensemble",
                "roi_proposals": [item[0] for item in self.ROI_PROPOSALS],
                "pose_guided_roi": False,
                "min_frame_support": self.min_frame_support,
                "min_vote_share": self.min_vote_share,
                "proposal_margin": self.proposal_margin,
                "license_note": "CC-BY-NC-3.0 research challenger",
            }
        )
        print(
            "🪪 Jersey challenger: parseq_soccernet_lit same PARSeq reader + "
            f"Hockey legibility + ROI/frame consensus device={self.device.type} "
            f"frames={self.max_frames} legibility>={self.legibility_threshold:.2f} "
            f"support>={self.min_frame_support} share>={self.min_vote_share:.2f}",
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

    def _ensure_legibility_checkpoint(self) -> None:
        if self.legibility_checkpoint.exists() and self.legibility_checkpoint.stat().st_size > 0:
            return
        if not self._env_bool("BASKETBALL_JERSEY_LEGIBILITY_AUTO_DOWNLOAD", True):
            raise FileNotFoundError(
                "Hockey jersey legibility checkpoint is missing: "
                f"{self.legibility_checkpoint}. Download Google Drive file "
                f"{HOCKEY_LEGIBILITY_GDRIVE_ID} there or enable "
                "BASKETBALL_JERSEY_LEGIBILITY_AUTO_DOWNLOAD."
            )
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "Literature PARSeq legibility auto-download requires gdown. "
                "Install the root requirements."
            ) from exc

        self.legibility_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = self.legibility_checkpoint.with_suffix(
            self.legibility_checkpoint.suffix + ".part"
        )
        if partial.exists():
            partial.unlink()
        print(
            "[PARSeq-lit] downloading Hockey legibility checkpoint to "
            f"{self.legibility_checkpoint}...",
            flush=True,
        )
        downloaded = gdown.download(
            id=HOCKEY_LEGIBILITY_GDRIVE_ID,
            output=str(partial),
            quiet=False,
        )
        if not downloaded or not partial.exists() or partial.stat().st_size <= 0:
            if partial.exists():
                partial.unlink()
            raise RuntimeError(
                "Could not download the Hockey jersey legibility checkpoint from Google Drive."
            )
        partial.replace(self.legibility_checkpoint)
        print(
            f"[PARSeq-lit] legibility checkpoint downloaded bytes={self.legibility_checkpoint.stat().st_size}",
            flush=True,
        )

    def _load_legibility_model(self) -> torch.nn.Module:
        started = time.perf_counter()
        print(
            f"[PARSeq-lit] loading Hockey ResNet34 legibility={self.legibility_checkpoint}...",
            flush=True,
        )
        raw = torch.load(
            str(self.legibility_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
            raw = raw["state_dict"]
        if not isinstance(raw, dict) or not raw:
            raise RuntimeError(
                f"Invalid legibility checkpoint: {self.legibility_checkpoint}"
            )

        state_dict: dict[str, Any] = {}
        for key, value in raw.items():
            clean = str(key)
            if clean.startswith("module."):
                clean = clean[len("module.") :]
            if clean.startswith("model_ft."):
                clean = clean[len("model_ft.") :]
            state_dict[clean] = value

        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Hockey legibility checkpoint is incompatible with torchvision ResNet34: "
                f"{exc}"
            ) from exc
        model = model.to(self.device).eval()
        print(
            f"[PARSeq-lit] legibility model ready in {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        return model

    @classmethod
    def _prepare_legibility(cls, crop: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            crop,
            (cls.LEGIBILITY_SIZE, cls.LEGIBILITY_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    def _legibility_scores(self, crops: list[np.ndarray]) -> list[float]:
        if not crops:
            return []
        batch = torch.stack([self._prepare_legibility(crop) for crop in crops]).to(self.device)
        with torch.inference_mode():
            logits = self.legibility_model(batch).reshape(-1)
            scores = torch.sigmoid(logits).float().detach().cpu().numpy()
        return [float(max(0.0, min(1.0, value))) for value in scores.tolist()]

    @classmethod
    def _roi_proposals(cls, player_crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
        if player_crop is None or player_crop.size == 0:
            return []
        height, width = player_crop.shape[:2]
        if height < 36 or width < 20:
            return []
        proposals: list[tuple[str, np.ndarray]] = []
        for name, y1f, y2f, x1f, x2f in cls.ROI_PROPOSALS:
            y1 = max(0, min(height - 1, int(round(height * y1f))))
            y2 = max(y1 + 1, min(height, int(round(height * y2f))))
            x1 = max(0, min(width - 1, int(round(width * x1f))))
            x2 = max(x1 + 1, min(width, int(round(width * x2f))))
            roi = player_crop[y1:y2, x1:x2]
            if roi.size and roi.shape[0] >= 12 and roi.shape[1] >= 10:
                proposals.append((name, roi))
        return proposals

    def _generate_batch(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not crops:
            return []
        batch = torch.stack([self._prepare_crop(crop) for crop in crops]).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            probabilities = logits.softmax(-1)
            predictions, confidences = self.model.tokenizer.decode(probabilities)

        outputs: list[tuple[str, float]] = []
        for index, prediction in enumerate(predictions):
            raw_output = str(prediction).strip()
            confidence = 0.0
            if index < len(confidences):
                token_confidence = confidences[index]
                if torch.is_tensor(token_confidence) and token_confidence.numel() > 0:
                    confidence = float(
                        token_confidence.float().mean().detach().cpu().item()
                    )
                elif token_confidence is not None:
                    try:
                        values = np.asarray(token_confidence, dtype=np.float32).reshape(-1)
                        if values.size:
                            confidence = float(values.mean())
                    except (TypeError, ValueError):
                        confidence = 0.0
            outputs.append((raw_output, max(0.0, min(1.0, confidence))))
        return outputs

    @staticmethod
    def _aggregate_frame_reads(
        frame_evidence: list[dict[str, Any]],
        *,
        min_confidence: float,
        min_frame_support: int,
        min_vote_share: float,
        proposal_margin: float,
    ) -> dict[str, Any]:
        """Collapse correlated ROI reads to one vote per frame, then aggregate frames."""
        frame_winners: list[dict[str, Any]] = []
        for frame in frame_evidence:
            best_by_number: dict[str, float] = {}
            for read in frame.get("reads", []):
                number = read.get("number")
                confidence = float(read.get("confidence", 0.0) or 0.0)
                if number is None or confidence < min_confidence:
                    continue
                best_by_number[str(number)] = max(
                    confidence,
                    float(best_by_number.get(str(number), 0.0)),
                )
            ranked = sorted(best_by_number.items(), key=lambda item: item[1], reverse=True)
            if not ranked:
                continue
            best_number, best_conf = ranked[0]
            runner_conf = ranked[1][1] if len(ranked) > 1 else 0.0
            if len(ranked) > 1 and best_conf - runner_conf < proposal_margin:
                continue
            frame_winners.append(
                {
                    "frame_id": frame.get("frame_id"),
                    "number": best_number,
                    "confidence": best_conf,
                    "legibility": frame.get("legibility"),
                    "crop_quality": frame.get("crop_quality"),
                }
            )

        if not frame_winners:
            return {
                "number": None,
                "reason": "no_valid_number",
                "confidence": 0.0,
                "margin": 0.0,
                "top_candidates": [],
                "frame_winners": [],
            }

        weights: dict[str, float] = defaultdict(float)
        support: dict[str, int] = defaultdict(int)
        confidences_by_number: dict[str, list[float]] = defaultdict(list)
        for winner in frame_winners:
            number = str(winner["number"])
            confidence = float(winner["confidence"])
            weights[number] += confidence
            support[number] += 1
            confidences_by_number[number].append(confidence)

        ranked_numbers = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        total_weight = max(sum(weights.values()), 1e-9)
        best_number, best_weight = ranked_numbers[0]
        second_weight = ranked_numbers[1][1] if len(ranked_numbers) > 1 else 0.0
        best_share = best_weight / total_weight
        second_share = second_weight / total_weight
        mean_best_conf = float(np.mean(confidences_by_number[best_number]))
        consensus_confidence = mean_best_conf * best_share
        top_candidates = [
            {
                "number": number,
                "posterior": round(weight / total_weight, 6),
                "support": int(support[number]),
                "weight": round(weight, 6),
            }
            for number, weight in ranked_numbers[:5]
        ]

        if support[best_number] < min_frame_support:
            reason = "low_visual_consensus_support"
            number: str | None = None
        elif best_share < min_vote_share:
            reason = "low_visual_consensus_share"
            number = None
        else:
            reason = "parsed_temporal_roi_consensus"
            number = best_number

        return {
            "number": number,
            "reason": reason,
            "confidence": float(max(0.0, min(1.0, consensus_confidence))),
            "margin": float(max(0.0, min(1.0, best_share - second_share))),
            "top_candidates": top_candidates,
            "frame_winners": frame_winners,
            "best_raw_number": best_number,
            "best_support": int(support[best_number]),
            "best_share": float(best_share),
        }

    def predict_jersey(
        self, track_id: int, team_name: str | None = None
    ) -> dict[str, Any]:
        selected = self._select_observations(int(track_id), max_images=self.max_frames)
        if not selected:
            return {
                "accepted": False,
                "rejection_reason": "crop_not_ready",
                "roster_policy": "validator_only",
            }

        started = time.perf_counter()
        crops = [item["crop"] for item in selected]
        legibility_scores = self._legibility_scores(crops)
        frame_evidence: list[dict[str, Any]] = []
        proposal_crops: list[np.ndarray] = []
        proposal_meta: list[tuple[int, str]] = []

        for index, (item, legibility) in enumerate(zip(selected, legibility_scores)):
            evidence = {
                "frame_id": item.get("frame_id"),
                "crop_quality": float(item.get("quality", 0.0)),
                "legibility": float(legibility),
                "reads": [],
            }
            frame_evidence.append(evidence)
            if legibility < self.legibility_threshold:
                continue
            for roi_name, roi in self._roi_proposals(item["crop"]):
                proposal_meta.append((index, roi_name))
                proposal_crops.append(roi)

        if proposal_crops:
            generated = self._generate_batch(proposal_crops)
            for (frame_index, roi_name), (raw_output, confidence) in zip(
                proposal_meta, generated
            ):
                number, parse_reason = self.parse_ocr_number(raw_output)
                frame_evidence[frame_index]["reads"].append(
                    {
                        "roi": roi_name,
                        "raw_output": raw_output,
                        "number": number,
                        "parse_reason": parse_reason,
                        "confidence": float(confidence),
                    }
                )

        aggregate = self._aggregate_frame_reads(
            frame_evidence,
            min_confidence=self.parseq_min_confidence,
            min_frame_support=self.min_frame_support,
            min_vote_share=self.min_vote_share,
            proposal_margin=self.proposal_margin,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        legible_selected = [
            item
            for item, score in zip(selected, legibility_scores)
            if score >= self.legibility_threshold
        ]
        selected_for_diagnostics = legible_selected or selected
        result = self._finalize_prediction(
            track_id=int(track_id),
            team_name=team_name,
            selected=selected_for_diagnostics,
            number=aggregate["number"],
            parse_reason=str(aggregate["reason"]),
            confidence=float(aggregate["confidence"]),
            margin=float(aggregate["margin"]),
            top_candidates=list(aggregate["top_candidates"]),
            latency_ms=latency_ms,
            raw_output=json.dumps(
                {
                    "best_raw_number": aggregate.get("best_raw_number"),
                    "best_support": aggregate.get("best_support", 0),
                    "best_share": round(float(aggregate.get("best_share", 0.0)), 6),
                },
                separators=(",", ":"),
            ),
            evidence_source="parseq_legibility_roi_frame_consensus",
        )
        result["candidate_count"] = len(aggregate.get("frame_winners", []))
        result["literature_support"] = {
            "frames_considered": len(selected),
            "frames_legible": len(legible_selected),
            "legibility_threshold": round(self.legibility_threshold, 6),
            "frame_winners": aggregate.get("frame_winners", []),
            "min_frame_support": self.min_frame_support,
            "min_vote_share": round(self.min_vote_share, 6),
            "proposal_margin": round(self.proposal_margin, 6),
            "roi_proposals": [item[0] for item in self.ROI_PROPOSALS],
        }
        result["per_crop_predictions"] = frame_evidence
        result["mean_crop_quality"] = round(
            float(np.mean([float(item.get("quality", 0.0)) for item in selected])), 6
        )
        return result
