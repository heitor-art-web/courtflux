from __future__ import annotations

import os
import re
import time
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
    """SoccerNet-finetuned PARSeq challenger for P1 jersey recognition.

    The checkpoint is the SoccerNet fine-tune released with Koshkina & Elder's
    CVPRW 2024 jersey-number pipeline. This adapter intentionally tests only the
    STR recognizer on CourtFlux's existing best torso crop; pose-guided number
    localization and the separate legibility classifier are not enabled in this
    first comparison. Raw OCR stays visual-only and the roster remains a validator.
    """

    BACKEND_NAME = "parseq_soccernet"
    INPUT_HEIGHT = 32
    INPUT_WIDTH = 128

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
        # The backend-specific confidence floor is deliberately conservative and
        # may be raised globally through BASKETBALL_JERSEY_CHALLENGER_MIN_CONFIDENCE.
        self.min_confidence = max(self.min_confidence, self.parseq_min_confidence)

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
                "SoccerNet PARSeq architecture load failed. Ensure the root "
                "requirements are installed and network access is available for "
                f"the first Torch Hub checkout: {type(exc).__name__}: {exc}"
            ) from exc

        print(f"[PARSeq] loading SoccerNet checkpoint={self.checkpoint_path}...", flush=True)
        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError(
                f"Invalid SoccerNet PARSeq checkpoint: missing state_dict in {self.checkpoint_path}"
            )

        # The released Lightning checkpoint stores PARSeq keys without the Torch
        # Hub wrapper's `model.` prefix. This is the compatibility mapping used by
        # recent Python 3.12/MPS consumers of the official checkpoint.
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
                "preprocessing": (
                    f"courtflux_{self.crop_mode}_crop_resize_32x128_rgb_minus1_plus1"
                ),
                "temporal_aggregation": "identity_v2_downstream",
                "roster_policy": "validator_only",
                "max_images_per_attempt": 1,
                "strict_output": "exact_1_or_2_digits_else_unknown",
                "min_read_confidence": self.parseq_min_confidence,
                "license_note": "CC-BY-NC-3.0 research challenger",
                "legibility_gate": False,
                "pose_guided_roi": False,
            }
        )
        print(
            f"🪪 Jersey challenger: {self.BACKEND_NAME} device={self.device.type} "
            f"crop={self.crop_mode} min_conf={self.parseq_min_confidence:.2f} "
            f"loaded_in={time.perf_counter() - started:.2f}s",
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
                "SoccerNet PARSeq checkpoint is missing: "
                f"{self.checkpoint_path}. Download Google Drive file "
                f"{PARSEQ_SOCCERNET_GDRIVE_ID} there or enable "
                "BASKETBALL_JERSEY_PARSEQ_AUTO_DOWNLOAD."
            )
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "SoccerNet PARSeq auto-download requires gdown. Install the root requirements."
            ) from exc

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".part")
        if partial.exists():
            partial.unlink()
        print(
            "[PARSeq] downloading official SoccerNet checkpoint to "
            f"{self.checkpoint_path}...",
            flush=True,
        )
        downloaded = gdown.download(
            id=PARSEQ_SOCCERNET_GDRIVE_ID,
            output=str(partial),
            quiet=False,
        )
        if not downloaded or not partial.exists() or partial.stat().st_size <= 0:
            if partial.exists():
                partial.unlink()
            raise RuntimeError(
                "Could not download the official SoccerNet PARSeq checkpoint from Google Drive."
            )
        partial.replace(self.checkpoint_path)
        print(
            f"[PARSeq] checkpoint downloaded bytes={self.checkpoint_path.stat().st_size}",
            flush=True,
        )

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

    def predict_jersey(
        self, track_id: int, team_name: str | None = None
    ) -> dict[str, Any]:
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
