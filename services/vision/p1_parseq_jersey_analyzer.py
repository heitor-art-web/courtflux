from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from services.vision.p1_jersey_challengers import ExternalJerseyBase


PARSEQ_SOCCERNET_GDRIVE_ID = "1uRln22tlhneVt3P6MePmVxBWSLMsL3bm"
PARSEQ_TORCH_HUB_REPO = "baudm/parseq:1902db043c029a7e03a3818c616c06600af574be"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "parseq_soccernet.ckpt"
DEFAULT_TORCH_HUB_DIR = PROJECT_ROOT / "weights" / "torch_hub"


class SoccerNetPARSeqJerseyAnalyzer(ExternalJerseyBase):
    """SoccerNet-finetuned PARSeq challenger for jersey-number research."""

    BACKEND_NAME = "parseq_soccernet"
    INPUT_HEIGHT = 32
    INPUT_WIDTH = 128
    RESEARCH_ROIS = (
        ("upper_center", 0.16, 0.04, 0.84, 0.48, 1.00),
        ("center", 0.12, 0.16, 0.88, 0.68, 1.00),
        ("lower_center", 0.16, 0.30, 0.84, 0.82, 0.92),
        ("full_torso", 0.00, 0.00, 1.00, 1.00, 0.62),
    )

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        super().__init__(profiling_path=profiling_path)

        requested_device = str(
            os.getenv("BASKETBALL_JERSEY_HF_DEVICE", device or "cpu") or "cpu"
        ).lower().strip()
        if requested_device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        configured_checkpoint = str(
            os.getenv("BASKETBALL_JERSEY_PARSEQ_CHECKPOINT", "") or ""
        ).strip()
        checkpoint_path = (
            Path(configured_checkpoint).expanduser()
            if configured_checkpoint
            else DEFAULT_CHECKPOINT
        )
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        self.checkpoint_path = checkpoint_path.resolve()
        self.model_path = str(self.checkpoint_path)

        self.parseq_min_confidence = self._env_float(
            "BASKETBALL_JERSEY_PARSEQ_MIN_CONFIDENCE", 0.50, 0.0, 1.0
        )
        self.min_confidence = max(self.min_confidence, self.parseq_min_confidence)
        self.research_multiframe = self._env_bool(
            "BASKETBALL_JERSEY_RESEARCH_MULTIFRAME", False
        )
        self.research_roi = self._env_bool(
            "BASKETBALL_JERSEY_RESEARCH_ROI", False
        )
        self.research_top_k = int(
            self._env_float("BASKETBALL_JERSEY_RESEARCH_TOP_K", 5.0, 2.0, 8.0)
        )
        self.research_min_support = int(
            self._env_float("BASKETBALL_JERSEY_RESEARCH_MIN_SUPPORT", 2.0, 2.0, 5.0)
        )
        self.research_min_share = self._env_float(
            "BASKETBALL_JERSEY_RESEARCH_MIN_SHARE", 0.60, 0.50, 0.95
        )

        self._ensure_checkpoint()

        torch_hub_dir = Path(
            os.getenv("BASKETBALL_JERSEY_PARSEQ_TORCH_HUB", str(DEFAULT_TORCH_HUB_DIR))
        ).expanduser()
        if not torch_hub_dir.is_absolute():
            torch_hub_dir = PROJECT_ROOT / torch_hub_dir
        torch_hub_dir = torch_hub_dir.resolve()
        torch_hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(torch_hub_dir))

        hub_repo = str(
            os.getenv("BASKETBALL_JERSEY_PARSEQ_HUB_REPO", PARSEQ_TORCH_HUB_REPO)
            or PARSEQ_TORCH_HUB_REPO
        ).strip()

        started = time.perf_counter()
        print(
            f"[PARSeq] loading architecture repo={hub_repo} hub={torch_hub_dir}...",
            flush=True,
        )
        try:
            model = torch.hub.load(
                hub_repo,
                "parseq",
                pretrained=False,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "SoccerNet PARSeq architecture load failed. Ensure the root requirements "
                f"are installed: {type(exc).__name__}: {exc}"
            ) from exc

        print(f"[PARSeq] loading SoccerNet checkpoint={self.checkpoint_path}...", flush=True)
        checkpoint = torch.load(
            str(self.checkpoint_path), map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(
                f"Invalid SoccerNet PARSeq checkpoint: missing state_dict in {self.checkpoint_path}"
            )
        mapped_state = {
            (key if str(key).startswith("model.") else f"model.{key}"): value
            for key, value in state_dict.items()
        }
        try:
            model.load_state_dict(mapped_state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "SoccerNet PARSeq checkpoint is incompatible with the pinned PARSeq "
                f"architecture: {exc}"
            ) from exc

        print(f"[PARSeq] checkpoint ready; moving model to {self.device.type}...", flush=True)
        self.model = model.to(self.device).eval()
        print("[PARSeq] model ready", flush=True)

        temporal = "identity_v2_downstream"
        if self.research_multiframe:
            temporal = (
                "research_multiframe_roi_then_identity_v2"
                if self.research_roi
                else "research_multiframe_then_identity_v2"
            )
        self.checkpoint_diagnostics.update(
            {
                "backend": self.BACKEND_NAME,
                "architecture": "parseq_soccernet_finetuned",
                "model_name": "Koshkina-Elder SoccerNet PARSeq",
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_source": "official_cvprw2024_google_drive",
                "checkpoint_gdrive_id": PARSEQ_SOCCERNET_GDRIVE_ID,
                "parseq_code_ref": hub_repo,
                "provider": "local_torch_hub",
                "device": self.device.type,
                "preprocessing": f"courtflux_{self.crop_mode}_crop_resize_32x128_rgb_minus1_plus1",
                "temporal_aggregation": temporal,
                "roster_policy": "validator_only",
                "max_images_per_attempt": self.research_top_k if self.research_multiframe else 1,
                "strict_output": "exact_1_or_2_digits_else_unknown",
                "min_read_confidence": self.parseq_min_confidence,
                "license_note": "CC-BY-NC-3.0 research challenger",
                "legibility_gate": self.research_multiframe,
                "pose_guided_roi": False,
                "localized_roi_proposals": self.research_roi,
                "research_roi_names": [row[0] for row in self.RESEARCH_ROIS] if self.research_roi else [],
                "research_multiframe": self.research_multiframe,
                "research_min_support": self.research_min_support,
                "research_min_share": self.research_min_share,
            }
        )
        print(
            f"🪪 Jersey challenger: {self.BACKEND_NAME} device={self.device.type} "
            f"crop={self.crop_mode} min_conf={self.parseq_min_confidence:.2f} "
            f"multiframe={self.research_multiframe} roi={self.research_roi} "
            f"top_k={self.research_top_k} loaded_in={time.perf_counter() - started:.2f}s",
            flush=True,
        )

    @staticmethod
    def _env_bool(name: str, default: bool = True) -> bool:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            return bool(default)
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _ensure_checkpoint(self) -> None:
        if self.checkpoint_path.exists() and self.checkpoint_path.stat().st_size > 0:
            return
        if not self._env_bool("BASKETBALL_JERSEY_PARSEQ_AUTO_DOWNLOAD", True):
            raise FileNotFoundError(
                f"SoccerNet PARSeq checkpoint is missing: {self.checkpoint_path}"
            )
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "SoccerNet PARSeq auto-download requires gdown. Install root requirements."
            ) from exc
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".part")
        if partial.exists():
            partial.unlink()
        downloaded = gdown.download(
            id=PARSEQ_SOCCERNET_GDRIVE_ID,
            output=str(partial),
            quiet=False,
        )
        if not downloaded or not partial.exists() or partial.stat().st_size <= 0:
            if partial.exists():
                partial.unlink()
            raise RuntimeError("Could not download the official SoccerNet PARSeq checkpoint")
        partial.replace(self.checkpoint_path)

    @staticmethod
    def parse_ocr_number(value: Any) -> tuple[str | None, str]:
        if value is None:
            return None, "no_text_output"
        text = str(value).strip()
        if not text:
            return None, "no_text_output"
        if not re.fullmatch(r"\d{1,2}", text):
            return None, "no_valid_number"
        number = int(text)
        if not 0 <= number <= 99:
            return None, "out_of_range_or_malformed"
        return ("00" if text == "00" else str(number)), "parsed"

    @classmethod
    def _prepare_crop(cls, crop: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(
            crop,
            (cls.INPUT_WIDTH, cls.INPUT_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        return tensor * 2.0 - 1.0

    def _generate(self, crop: np.ndarray) -> tuple[str, float]:
        image = self._prepare_crop(crop).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(image)
            probabilities = logits.softmax(-1)
            predictions, confidences = self.model.tokenizer.decode(probabilities)
        raw_output = str(predictions[0]).strip() if predictions else ""
        confidence = 0.0
        if confidences:
            token_confidence = confidences[0]
            if torch.is_tensor(token_confidence) and token_confidence.numel() > 0:
                confidence = float(token_confidence.float().mean().detach().cpu().item())
            elif token_confidence is not None:
                try:
                    values = np.asarray(token_confidence, dtype=np.float32).reshape(-1)
                    if values.size:
                        confidence = float(values.mean())
                except (TypeError, ValueError):
                    confidence = 0.0
        return raw_output, max(0.0, min(1.0, confidence))

    @staticmethod
    def _research_legibility_score(crop: np.ndarray, base_quality: float) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if height < 24 or width < 16:
            return 0.0
        y1, y2 = int(height * 0.18), int(height * 0.88)
        x1, x2 = int(width * 0.16), int(width * 0.84)
        center = gray[y1:y2, x1:x2]
        if center.size == 0:
            return 0.0
        contrast = min(float(np.std(center)) / 62.0, 1.0)
        exposure = float(np.mean((center >= 28) & (center <= 230)))
        edges = cv2.Canny(center, 60, 150)
        edge_density = float(np.mean(edges > 0))
        edge_score = min(edge_density / 0.14, 1.0)
        resolution = min(float(center.size) / float(88 * 44), 1.0)
        score = (
            0.30 * max(0.0, min(1.0, float(base_quality)))
            + 0.25 * edge_score
            + 0.20 * contrast
            + 0.15 * exposure
            + 0.10 * resolution
        )
        return max(0.0, min(1.0, float(score)))

    @staticmethod
    def _roi_crop(
        crop: np.ndarray,
        x1f: float,
        y1f: float,
        x2f: float,
        y2f: float,
    ) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        height, width = crop.shape[:2]
        x1 = max(0, min(width - 1, int(round(width * x1f))))
        x2 = max(1, min(width, int(round(width * x2f))))
        y1 = max(0, min(height - 1, int(round(height * y1f))))
        y2 = max(1, min(height, int(round(height * y2f))))
        if x2 - x1 < 12 or y2 - y1 < 12:
            return None
        result = crop[y1:y2, x1:x2]
        return result if result.size else None

    def _research_roi_proposals(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        crop = item.get("crop")
        if crop is None or crop.size == 0:
            return []
        proposals: list[dict[str, Any]] = []
        for name, x1, y1, x2, y2, prior in self.RESEARCH_ROIS:
            roi = self._roi_crop(crop, x1, y1, x2, y2)
            if roi is None:
                continue
            roi_quality = self._research_legibility_score(
                roi, float(item.get("quality", 0.0))
            )
            proposals.append(
                {
                    "roi_name": name,
                    "roi": roi,
                    "roi_prior": float(prior),
                    "roi_legibility_score": roi_quality,
                }
            )
        return proposals

    def _research_select_observations(self, track_id: int) -> list[dict[str, Any]]:
        history = list(self.track_crop_buffers.get(int(track_id), []))
        if not history:
            return []
        last_used = int(self.track_last_inference_ordinal.get(int(track_id), 0))
        fresh = [item for item in history if int(item.get("ordinal", 0)) > last_used]
        pool = fresh if len(fresh) >= self.research_min_support else history
        ranked: list[dict[str, Any]] = []
        for item in pool:
            copy = dict(item)
            copy["legibility_score"] = self._research_legibility_score(
                copy["crop"], float(copy.get("quality", 0.0))
            )
            ranked.append(copy)
        selected = sorted(
            ranked,
            key=lambda item: (
                float(item.get("legibility_score", 0.0)),
                float(item.get("quality", 0.0)),
                int(item.get("ordinal", 0)),
            ),
            reverse=True,
        )[: self.research_top_k]
        selected.sort(key=lambda item: int(item.get("ordinal", 0)))
        if selected:
            newest = max(int(item.get("ordinal", 0)) for item in selected)
            self.track_last_inference_ordinal[int(track_id)] = max(last_used, newest)
        return selected

    @staticmethod
    def _multi_frame_consensus(
        rows: list[dict[str, Any]], *, min_support: int, min_share: float
    ) -> dict[str, Any]:
        weights: dict[str, float] = defaultdict(float)
        frame_support: dict[str, set[int]] = defaultdict(set)
        confidence_sums: dict[str, float] = defaultdict(float)
        confidence_counts: dict[str, int] = defaultdict(int)
        valid_weight = 0.0
        for row in rows:
            number = row.get("number")
            confidence = max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0)))
            legibility = max(0.0, min(1.0, float(row.get("legibility_score", 0.0) or 0.0)))
            roi_prior = max(0.0, min(1.0, float(row.get("roi_prior", 1.0) or 1.0)))
            if number is None:
                continue
            weight = max(0.02, confidence * max(0.20, legibility) * roi_prior)
            key = str(number)
            weights[key] += weight
            frame_id = row.get("frame_id")
            if isinstance(frame_id, int):
                frame_support[key].add(frame_id)
            confidence_sums[key] += confidence
            confidence_counts[key] += 1
            valid_weight += weight
        if not weights or valid_weight <= 0.0:
            return {
                "number": None,
                "reason": "no_valid_number",
                "support": 0,
                "share": 0.0,
                "margin": 0.0,
                "top_candidates": [],
            }
        ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        top_number, top_weight = ordered[0]
        second_weight = ordered[1][1] if len(ordered) > 1 else 0.0
        share = top_weight / valid_weight
        margin = (top_weight - second_weight) / valid_weight
        support = len(frame_support[top_number])
        top_candidates = [
            {
                "number": number,
                "posterior": round(weight / valid_weight, 6),
                "support": len(frame_support[number]),
            }
            for number, weight in ordered[:5]
        ]
        if support < int(min_support) or share < float(min_share):
            return {
                "number": None,
                "reason": "low_multiframe_consensus",
                "support": support,
                "share": share,
                "margin": margin,
                "top_candidates": top_candidates,
                "best_candidate": top_number,
            }
        return {
            "number": top_number,
            "reason": "parsed",
            "support": support,
            "share": share,
            "margin": margin,
            "mean_confidence": confidence_sums[top_number]
            / max(confidence_counts[top_number], 1),
            "top_candidates": top_candidates,
        }

    def predict_jersey(
        self, track_id: int, team_name: str | None = None
    ) -> dict[str, Any]:
        if not self.research_multiframe:
            selected = self._select_observations(int(track_id), max_images=1)
            if not selected:
                return {
                    "accepted": False,
                    "rejection_reason": "crop_not_ready",
                    "roster_policy": "validator_only",
                }
            started = time.perf_counter()
            try:
                raw_output, confidence = self._generate(selected[0]["crop"])
            except Exception as exc:
                raise RuntimeError(
                    f"SoccerNet PARSeq jersey inference failed: {type(exc).__name__}: {exc}"
                ) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0
            number, parse_reason = self.parse_ocr_number(raw_output)
            return self._finalize_prediction(
                track_id=int(track_id),
                team_name=team_name,
                selected=selected,
                number=number,
                parse_reason=parse_reason,
                confidence=confidence,
                margin=0.0,
                latency_ms=latency_ms,
                raw_output=raw_output,
                evidence_source="parseq_mean_token_confidence",
            )

        selected = self._research_select_observations(int(track_id))
        if not selected:
            return {
                "accepted": False,
                "rejection_reason": "crop_not_ready",
                "roster_policy": "validator_only",
            }

        started = time.perf_counter()
        per_crop: list[dict[str, Any]] = []
        try:
            for item in selected:
                if self.research_roi:
                    proposals = self._research_roi_proposals(item)
                else:
                    proposals = [
                        {
                            "roi_name": "full_torso",
                            "roi": item["crop"],
                            "roi_prior": 1.0,
                            "roi_legibility_score": float(item.get("legibility_score", 0.0)),
                        }
                    ]
                for proposal in proposals:
                    raw_output, confidence = self._generate(proposal["roi"])
                    number, parse_reason = self.parse_ocr_number(raw_output)
                    per_crop.append(
                        {
                            "frame_id": item.get("frame_id"),
                            "crop_quality": round(float(item.get("quality", 0.0)), 6),
                            "legibility_score": round(
                                float(proposal.get("roi_legibility_score", 0.0)), 6
                            ),
                            "roi_name": proposal.get("roi_name"),
                            "roi_prior": round(float(proposal.get("roi_prior", 1.0)), 6),
                            "raw_model_output": raw_output,
                            "raw_canonical": number,
                            "parse_reason": parse_reason,
                            "confidence": round(float(confidence), 6),
                            "number": number,
                        }
                    )
        except Exception as exc:
            raise RuntimeError(
                f"SoccerNet PARSeq jersey inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        consensus = self._multi_frame_consensus(
            per_crop,
            min_support=self.research_min_support,
            min_share=self.research_min_share,
        )
        number = consensus.get("number")
        confidence = float(consensus.get("share", 0.0) or 0.0)
        margin = float(consensus.get("margin", 0.0) or 0.0)
        prediction = self._finalize_prediction(
            track_id=int(track_id),
            team_name=team_name,
            selected=selected,
            number=str(number) if number is not None else None,
            parse_reason=str(consensus.get("reason") or "no_valid_number"),
            confidence=confidence,
            margin=margin,
            top_candidates=list(consensus.get("top_candidates") or []),
            latency_ms=latency_ms,
            raw_output="|".join(str(row.get("raw_model_output") or "") for row in per_crop),
            evidence_source=(
                "parseq_multiframe_roi_consensus"
                if self.research_roi
                else "parseq_multiframe_legibility_weighted_consensus"
            ),
        )
        prediction["candidate_count"] = int(consensus.get("support", 0) or 0)
        prediction["per_crop_predictions"] = per_crop
        prediction["multiframe_consensus"] = {
            "support": int(consensus.get("support", 0) or 0),
            "share": round(confidence, 6),
            "margin": round(margin, 6),
            "min_support": self.research_min_support,
            "min_share": self.research_min_share,
            "selected_images": len(selected),
            "roi_proposals_per_image": len(self.RESEARCH_ROIS) if self.research_roi else 1,
            "roi_enabled": self.research_roi,
        }
        prediction["mean_legibility_score"] = round(
            float(np.mean([row["legibility_score"] for row in per_crop])) if per_crop else 0.0,
            6,
        )
        return prediction
