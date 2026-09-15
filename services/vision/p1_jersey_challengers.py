from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


ROBOFLOW_CLASSIFIER_MODEL = "basketball-jersey-numbers-class-s8khj/3"
EBQWEN_MODEL = "GabrieleGiudici/EBQwen2.5-VL-3B"
SMOLVLM2_JERSEY_MODEL = "Utkarsh-HF/smolvlm2-jersey-ocr-merged"

STRICT_PROMPT = (
    "Read the basketball player's jersey number. Return only one value: the number from "
    "0 to 99, or UNKNOWN if it is not readable. Do not guess from a roster, player name, "
    "team, or context."
)


class ExternalJerseyBase(TeamOnlyAnalyzer):
    """Common CourtFlux adapter for experimental external jersey recognizers.

    External challengers provide only raw visual evidence. Team rosters validate the raw
    number after inference and never rewrite or select a different number. Identity-v2
    remains responsible for cross-attempt temporal locking.
    """

    BACKEND_NAME = "external"

    def __init__(
        self,
        profiling_path: str,
        *,
        crop_buffer_size: int = 16,
        crop_mode: str | None = None,
    ) -> None:
        super().__init__(profiling_path=profiling_path)
        self.crop_buffer_size = max(4, min(int(crop_buffer_size), 48))
        requested_crop = str(
            crop_mode
            or os.getenv("BASKETBALL_JERSEY_CHALLENGER_CROP", "torso")
            or "torso"
        ).strip().lower()
        if requested_crop not in {"torso", "player"}:
            raise ValueError("BASKETBALL_JERSEY_CHALLENGER_CROP must be torso or player")
        self.crop_mode = requested_crop
        self.min_confidence = self._env_float(
            "BASKETBALL_JERSEY_CHALLENGER_MIN_CONFIDENCE", 0.0, 0.0, 1.0
        )
        self.track_crop_buffers: dict[int, list[dict[str, Any]]] = {}
        self.track_observation_index: dict[int, int] = {}
        self.track_last_inference_ordinal: dict[int, int] = {}
        self.checkpoint_diagnostics: dict[str, Any] = {
            "backend": self.BACKEND_NAME,
            "roster_policy": "validator_only",
            "temporal_aggregation": "identity_v2_downstream",
            "experimental_challenger": True,
        }

    @staticmethod
    def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
        raw = os.getenv(name)
        try:
            value = float(raw) if raw is not None and raw.strip() else float(default)
        except (TypeError, ValueError):
            value = float(default)
        return max(minimum, min(maximum, value))

    @staticmethod
    def _torso_crop(crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        height, width = crop.shape[:2]
        if height < 36 or width < 20:
            return None
        # Wider than the legacy digit-head torso crop. The external models were trained
        # on player/jersey imagery and benefit from retaining shoulders + lower jersey.
        y1, y2 = int(height * 0.06), int(height * 0.76)
        x1, x2 = int(width * 0.06), int(width * 0.94)
        focused = crop[y1:y2, x1:x2]
        return focused if focused.size else None

    def _focus_crop(self, crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        if self.crop_mode == "player":
            return crop
        return self._torso_crop(crop)

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

    def observe_track(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_id: int | None = None,
    ) -> None:
        track_id = int(track_id)
        if track_id < 0:
            return
        focused = self._focus_crop(crop)
        if focused is None:
            return
        ordinal = int(self.track_observation_index.get(track_id, 0)) + 1
        self.track_observation_index[track_id] = ordinal
        buffer = list(self.track_crop_buffers.get(track_id, []))
        buffer.append(
            {
                "crop": focused.copy(),
                "quality": self._crop_quality(focused),
                "ordinal": ordinal,
                "frame_id": frame_id,
            }
        )
        self.track_crop_buffers[track_id] = buffer[-self.crop_buffer_size :]
        if len(self.track_crop_buffers) > 250:
            keep = sorted(
                self.track_crop_buffers,
                key=lambda key: int(self.track_observation_index.get(key, 0)),
                reverse=True,
            )[:120]
            self.track_crop_buffers = {
                key: self.track_crop_buffers[key] for key in keep
            }
            self.track_observation_index = {
                key: self.track_observation_index.get(key, 0) for key in keep
            }
            self.track_last_inference_ordinal = {
                key: self.track_last_inference_ordinal.get(key, 0) for key in keep
            }

    def ready(self, track_id: int) -> bool:
        return bool(self.track_crop_buffers.get(int(track_id)))

    def _select_observations(self, track_id: int, max_images: int = 1) -> list[dict[str, Any]]:
        history = list(self.track_crop_buffers.get(int(track_id), []))
        if not history:
            return []
        last_used = int(self.track_last_inference_ordinal.get(int(track_id), 0))
        fresh = [item for item in history if int(item.get("ordinal", 0)) > last_used]
        pool = fresh if fresh else history
        count = max(1, min(int(max_images), len(pool)))
        selected = sorted(
            pool,
            key=lambda item: (float(item.get("quality", 0.0)), int(item.get("ordinal", 0))),
            reverse=True,
        )[:count]
        selected.sort(key=lambda item: int(item.get("ordinal", 0)))
        newest = max(int(item.get("ordinal", 0)) for item in selected)
        self.track_last_inference_ordinal[int(track_id)] = max(last_used, newest)
        return selected

    @classmethod
    def parse_number(cls, value: Any) -> tuple[str | None, str]:
        """Extract exactly one raw 0-99 number without consulting a roster."""
        if value is None:
            return None, "no_text_output"
        text = str(value).strip()
        if not text:
            return None, "no_text_output"
        if text.upper() in {"UNKNOWN", "UNREADABLE", "NONE", "NAN", "N/A", "NA"}:
            return None, "model_abstained"
        if re.search(r"(?<!\d)\d{3,}(?!\d)", text):
            return None, "out_of_range_or_malformed"
        tokens = re.findall(r"(?<!\d)\d{1,2}(?!\d)", text)
        candidates: set[str] = set()
        for token in tokens:
            number = int(token)
            if 0 <= number <= 99:
                candidates.add("00" if token == "00" else str(number))
        if len(candidates) == 1:
            return next(iter(candidates)), "parsed"
        if len(candidates) > 1:
            return None, "ambiguous_numeric_output"
        return None, "no_valid_number"

    def _valid_roster_numbers(self, team_name: str | None) -> list[str]:
        if not team_name or team_name not in self.rosters:
            return []
        roster = (self.rosters.get(team_name, {}) or {}).get("roster", {}) or {}
        values: list[str] = []
        for key in roster:
            text = str(key).strip()
            if text == "00":
                values.append("00")
            elif text.isdigit() and 0 <= int(text) <= 99:
                values.append(str(int(text)))
        return sorted(set(values), key=lambda item: -1 if item == "00" else int(item))

    def _finalize_prediction(
        self,
        *,
        track_id: int,
        team_name: str | None,
        selected: list[dict[str, Any]],
        number: str | None,
        parse_reason: str,
        confidence: float,
        margin: float = 0.0,
        top_candidates: list[dict[str, Any]] | None = None,
        latency_ms: float | None = None,
        raw_output: str | None = None,
        evidence_source: str = "model_confidence",
    ) -> dict[str, Any]:
        confidence = max(0.0, min(1.0, float(confidence)))
        margin = max(0.0, min(1.0, float(margin)))
        valid = self._valid_roster_numbers(team_name)
        roster_valid = bool(number is not None and number in valid)

        if number is None:
            accepted, reason = False, parse_reason
        elif confidence < self.min_confidence:
            accepted, reason = False, "low_challenger_confidence"
        elif not team_name:
            accepted, reason = False, "team_unknown_for_roster_validation"
        elif not valid:
            accepted, reason = False, "no_known_roster_numbers"
        elif not roster_valid:
            accepted, reason = False, "raw_not_in_roster"
        else:
            accepted, reason = True, "accepted"

        cross_team_validation: dict[str, Any] = {}
        for candidate_team in self.rosters:
            team_valid = self._valid_roster_numbers(str(candidate_team))
            is_valid = bool(number is not None and number in team_valid)
            cross_team_validation[str(candidate_team)] = {
                "candidate_number": number if is_valid else None,
                "candidate_posterior": round(confidence, 6) if is_valid else 0.0,
                "candidate_margin": round(margin, 6) if is_valid else 0.0,
                "roster_valid": is_valid,
            }

        qualities = [round(float(item.get("quality", 0.0)), 6) for item in selected]
        if top_candidates is None:
            top_candidates = (
                [{"number": number, "posterior": round(confidence, 6)}]
                if number is not None
                else []
            )
        return {
            "accepted": accepted,
            "rejection_reason": reason,
            "team": team_name,
            "candidate_number": number,
            "raw_number": number,
            "raw_canonical": number,
            "raw_decode_reason": parse_reason,
            "candidate_probability": round(confidence, 6),
            "candidate_posterior": round(confidence, 6),
            "candidate_margin": round(margin, 6),
            "candidate_count": len(selected),
            "candidate_evidence": round(confidence, 6),
            "candidate_evidence_source": evidence_source,
            "top_candidates": top_candidates,
            "valid_roster_numbers": valid,
            "roster_valid": roster_valid,
            "roster_policy": "validator_only",
            "cross_team_validation": cross_team_validation,
            "cross_team_best": cross_team_validation,
            "crop_quality": qualities,
            "mean_crop_quality": round(float(np.mean(qualities)), 6) if qualities else 0.0,
            "per_crop_predictions": [
                {
                    "frame_id": item.get("frame_id"),
                    "crop_quality": round(float(item.get("quality", 0.0)), 6),
                    "challenger_input": True,
                }
                for item in selected
            ],
            "latency_ms": round(float(latency_ms), 3) if latency_ms is not None else None,
            "raw_model_output": raw_output,
            "checkpoint": self.checkpoint_diagnostics,
            "p1_raw_roster_consistency_gate": True,
        }


class RoboflowClassificationJerseyAnalyzer(ExternalJerseyBase):
    """Hosted Roboflow ResNet50 classification challenger.

    The selected public model exposes 42 jersey classes and includes CourtFlux-relevant
    values such as 13, 21, 22 and 23. Its reported Universe metrics are not treated as
    CourtFlux accuracy; only integrated run metrics are used for this challenger.
    """

    BACKEND_NAME = "roboflow_classification"

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        super().__init__(profiling_path=profiling_path)
        del device  # hosted inference; kept for backend constructor compatibility
        self.api_url = str(
            os.getenv("ROBOFLOW_JERSEY_API_URL", "https://serverless.roboflow.com")
        ).rstrip("/")
        self.model_id = str(
            os.getenv("ROBOFLOW_JERSEY_CLASS_MODEL_ID", ROBOFLOW_CLASSIFIER_MODEL)
        ).strip()
        self.model_path = self.model_id
        api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is required for BASKETBALL_JERSEY_BACKEND="
                "roboflow_classification"
            )
        try:
            from inference_sdk import InferenceConfiguration, InferenceHTTPClient
        except ImportError as exc:
            raise RuntimeError(
                "Roboflow challenger requires inference-sdk. Install the root requirements."
            ) from exc
        client = InferenceHTTPClient(api_url=self.api_url, api_key=api_key)
        try:
            client = client.configure(InferenceConfiguration(api_key_transport="header"))
        except Exception:
            pass
        self.client = client
        self.checkpoint_diagnostics.update(
            {
                "backend": self.BACKEND_NAME,
                "architecture": "roboflow_resnet50_classification",
                "model_name": self.model_id,
                "checkpoint": self.model_id,
                "provider": "roboflow_serverless",
                "preprocessing": f"courtflux_{self.crop_mode}_crop_provider_managed",
                "temporal_aggregation": "identity_v2_downstream",
                "roster_policy": "validator_only",
            }
        )
        print(
            "🪪 Jersey challenger: Roboflow classification "
            f"model={self.model_id} api={self.api_url} crop={self.crop_mode}",
            flush=True,
        )

    @classmethod
    def _classification_candidates(cls, response: Any) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        payloads = response if isinstance(response, list) else [response]
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            predictions = payload.get("predictions")
            if isinstance(predictions, dict):
                predictions = [
                    {"class": key, "confidence": value}
                    for key, value in predictions.items()
                ]
            if isinstance(predictions, list):
                for item in predictions:
                    if not isinstance(item, dict):
                        continue
                    label = item.get("class", item.get("class_name", item.get("label")))
                    confidence = item.get("confidence", item.get("score", 0.0))
                    number, reason = cls.parse_number(label)
                    if number is None or reason != "parsed":
                        continue
                    try:
                        score = float(confidence)
                    except (TypeError, ValueError):
                        score = 0.0
                    candidates.append({"number": number, "posterior": score})
            top = payload.get("top")
            if top is not None and not candidates:
                number, reason = cls.parse_number(top)
                if number is not None and reason == "parsed":
                    try:
                        score = float(payload.get("confidence", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0
                    candidates.append({"number": number, "posterior": score})
        deduped: dict[str, float] = {}
        for item in candidates:
            number = str(item["number"])
            deduped[number] = max(deduped.get(number, 0.0), float(item["posterior"]))
        return sorted(
            [
                {"number": number, "posterior": round(score, 6)}
                for number, score in deduped.items()
            ],
            key=lambda item: float(item["posterior"]),
            reverse=True,
        )

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
        crop = selected[0]["crop"]
        started = time.perf_counter()
        try:
            response = self.client.infer(crop, model_id=self.model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Roboflow jersey inference failed for {self.model_id}: {type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        candidates = self._classification_candidates(response)
        if candidates:
            number = str(candidates[0]["number"])
            confidence = float(candidates[0]["posterior"])
            second = float(candidates[1]["posterior"]) if len(candidates) > 1 else 0.0
            margin = max(0.0, confidence - second)
            parse_reason = "parsed"
        else:
            number, confidence, margin = None, 0.0, 0.0
            parse_reason = "no_valid_number"
        return self._finalize_prediction(
            track_id=int(track_id),
            team_name=team_name,
            selected=selected,
            number=number,
            parse_reason=parse_reason,
            confidence=confidence,
            margin=margin,
            top_candidates=candidates[:5],
            latency_ms=latency_ms,
            raw_output=None,
            evidence_source="roboflow_classification_confidence",
        )


class HuggingFaceVLMJerseyAnalyzer(ExternalJerseyBase):
    """Local Hugging Face image-to-text challenger with strict numeric decoding."""

    def __init__(
        self,
        profiling_path: str,
        *,
        device: str,
        model_id: str,
        backend_name: str,
        max_images: int,
        prompt: str = STRICT_PROMPT,
    ) -> None:
        self.BACKEND_NAME = backend_name
        super().__init__(profiling_path=profiling_path)
        self.model_id = str(model_id).strip()
        self.model_path = self.model_id
        self.max_images = max(1, min(int(max_images), 8))
        self.prompt = str(prompt)
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
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as AutoVisionTextModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVisionTextModel
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face jersey challengers require transformers + accelerate. "
                "Install the root requirements."
            ) from exc

        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device.type in {"mps", "cuda"} else torch.float32
        load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        try:
            self.model = AutoVisionTextModel.from_pretrained(
                self.model_id,
                dtype=dtype,
                **load_kwargs,
            )
        except TypeError:
            self.model = AutoVisionTextModel.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                **load_kwargs,
            )
        self.model = self.model.to(self.device).eval()
        self.checkpoint_diagnostics.update(
            {
                "backend": backend_name,
                "architecture": "vision_language_generation",
                "model_name": self.model_id,
                "checkpoint": self.model_id,
                "provider": "huggingface_local",
                "device": self.device.type,
                "preprocessing": f"courtflux_{self.crop_mode}_crop_auto_processor",
                "temporal_aggregation": "identity_v2_downstream",
                "roster_policy": "validator_only",
                "max_images_per_attempt": self.max_images,
            }
        )
        print(
            f"🪪 Jersey challenger: {backend_name} model={self.model_id} "
            f"device={self.device.type} images/attempt={self.max_images} "
            f"loaded_in={time.perf_counter() - started:.2f}s",
            flush=True,
        )

    def _generate(self, images: list[np.ndarray]) -> tuple[str, float]:
        pil_images = [
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images
        ]
        content: list[dict[str, Any]] = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": self.prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
        )
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        input_length = int(moved["input_ids"].shape[1]) if "input_ids" in moved else 0
        with torch.inference_mode():
            output = self.model.generate(
                **moved,
                max_new_tokens=8,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequence = output.sequences[0]
        generated = sequence[input_length:]
        decoded = self.processor.batch_decode(
            generated.unsqueeze(0),
            skip_special_tokens=True,
        )[0].strip()

        log_probs: list[float] = []
        scores = list(getattr(output, "scores", []) or [])
        for step, logits in enumerate(scores[: int(generated.numel())]):
            token_id = int(generated[step].item())
            token_log_prob = torch.log_softmax(logits[0].float(), dim=-1)[token_id]
            log_probs.append(float(token_log_prob.detach().cpu().item()))
        confidence = (
            math.exp(sum(log_probs) / len(log_probs)) if log_probs else 0.0
        )
        return decoded, max(0.0, min(1.0, float(confidence)))

    def predict_jersey(
        self, track_id: int, team_name: str | None = None
    ) -> dict[str, Any]:
        selected = self._select_observations(
            int(track_id), max_images=self.max_images
        )
        if not selected:
            return {
                "accepted": False,
                "rejection_reason": "crop_not_ready",
                "roster_policy": "validator_only",
            }
        started = time.perf_counter()
        try:
            raw_output, confidence = self._generate(
                [item["crop"] for item in selected]
            )
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face jersey inference failed for {self.model_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        number, parse_reason = self.parse_number(raw_output)
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
            evidence_source="generation_sequence_confidence",
        )


class EBQwenJerseyAnalyzer(HuggingFaceVLMJerseyAnalyzer):
    BACKEND_NAME = "ebqwen_vl"

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        model_id = os.getenv("BASKETBALL_JERSEY_EBQWEN_MODEL", EBQWEN_MODEL)
        prompt = (
            "These images show the same tracked basketball player. Read the jersey number "
            "using only visible digits across the images. Return exactly one value: 0 to "
            "99, or UNKNOWN if unreadable. Do not infer from roster, team, or player identity."
        )
        super().__init__(
            profiling_path,
            device=device,
            model_id=model_id,
            backend_name=self.BACKEND_NAME,
            max_images=4,
            prompt=prompt,
        )


class SmolVLM2JerseyAnalyzer(HuggingFaceVLMJerseyAnalyzer):
    BACKEND_NAME = "smolvlm2_jersey"

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        model_id = os.getenv("BASKETBALL_JERSEY_SMOLVLM2_MODEL", SMOLVLM2_JERSEY_MODEL)
        super().__init__(
            profiling_path,
            device=device,
            model_id=model_id,
            backend_name=self.BACKEND_NAME,
            max_images=1,
            prompt=STRICT_PROMPT,
        )
