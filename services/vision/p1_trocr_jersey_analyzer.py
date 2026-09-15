from __future__ import annotations

import math
import os
import re
import time
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from services.vision.p1_jersey_challengers import ExternalJerseyBase


TROCR_SMALL_PRINTED_MODEL = "microsoft/trocr-small-printed"


class TrOCRSmallPrintedJerseyAnalyzer(ExternalJerseyBase):
    """Lightweight generic OCR challenger for P1 jersey-number recognition.

    TrOCR provides raw visual text only. CourtFlux accepts only an exact one- or
    two-digit OCR result, then applies the existing roster as a validator only.
    Identity v2 remains responsible for temporal evidence and reversible locks.
    """

    BACKEND_NAME = "trocr_small_printed"

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        super().__init__(profiling_path=profiling_path)
        self.model_id = str(
            os.getenv("BASKETBALL_JERSEY_TROCR_MODEL", TROCR_SMALL_PRINTED_MODEL)
        ).strip()
        self.model_path = self.model_id

        requested_device = str(
            os.getenv("BASKETBALL_JERSEY_HF_DEVICE", device or "cpu") or "cpu"
        ).lower().strip()
        if requested_device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif requested_device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError(
                "TrOCR challenger requires transformers. Install the root requirements."
            ) from exc

        started = time.perf_counter()
        print(
            f"[TrOCR] loading processor model={self.model_id}...",
            flush=True,
        )
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        except Exception:
            # Current Transformers normally supports TrOCRProcessor for this model,
            # but keep AutoProcessor as a compatibility fallback across releases.
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("[TrOCR] processor ready; loading model weights...", flush=True)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
        )
        print(f"[TrOCR] model weights ready; moving model to {self.device.type}...", flush=True)
        self.model = self.model.to(self.device).eval()
        print("[TrOCR] model ready", flush=True)

        self.checkpoint_diagnostics.update(
            {
                "backend": self.BACKEND_NAME,
                "architecture": "trocr_vision_encoder_decoder",
                "model_name": self.model_id,
                "checkpoint": self.model_id,
                "provider": "huggingface_local",
                "device": self.device.type,
                "preprocessing": f"courtflux_{self.crop_mode}_crop_trocr_processor",
                "temporal_aggregation": "identity_v2_downstream",
                "roster_policy": "validator_only",
                "max_images_per_attempt": 1,
                "strict_output": "exact_1_or_2_digits_else_unknown",
            }
        )
        print(
            f"🪪 Jersey challenger: {self.BACKEND_NAME} model={self.model_id} "
            f"device={self.device.type} crop={self.crop_mode} "
            f"loaded_in={time.perf_counter() - started:.2f}s",
            flush=True,
        )

    @staticmethod
    def parse_ocr_number(value: Any) -> tuple[str | None, str]:
        """Accept only an exact 0-99 OCR string; never extract digits from prose."""
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

    @staticmethod
    def _sequence_confidence(
        sequence: torch.Tensor,
        scores: list[torch.Tensor],
        special_token_ids: set[int],
    ) -> float:
        """Geometric mean token probability, excluding decoder special tokens."""
        if not scores or sequence.numel() == 0:
            return 0.0
        generated = sequence[-len(scores) :]
        log_probs: list[float] = []
        for step, logits in enumerate(scores[: int(generated.numel())]):
            token_id = int(generated[step].item())
            if token_id in special_token_ids:
                continue
            token_log_prob = torch.log_softmax(logits[0].float(), dim=-1)[token_id]
            log_probs.append(float(token_log_prob.detach().cpu().item()))
        if not log_probs:
            return 0.0
        confidence = math.exp(sum(log_probs) / len(log_probs))
        return max(0.0, min(1.0, float(confidence)))

    def _generate(self, image: np.ndarray) -> tuple[str, float]:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                pixel_values,
                max_new_tokens=6,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequence = output.sequences[0]
        raw_output = self.processor.batch_decode(
            output.sequences,
            skip_special_tokens=True,
        )[0].strip()
        tokenizer = getattr(self.processor, "tokenizer", None)
        special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        confidence = self._sequence_confidence(
            sequence,
            list(getattr(output, "scores", []) or []),
            special_token_ids,
        )
        return raw_output, confidence

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
                f"TrOCR jersey inference failed for {self.model_id}: "
                f"{type(exc).__name__}: {exc}"
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
            evidence_source="trocr_generation_token_confidence",
        )
