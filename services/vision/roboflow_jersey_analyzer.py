from __future__ import annotations

import os
import re
from typing import Any

import cv2
import numpy as np

from services.vision.team_only_analyzer import TeamOnlyAnalyzer


DEFAULT_JERSEY_MODEL_ID = "basketball-jersey-numbers-ocr/7"
DEFAULT_JERSEY_API_URL = "http://localhost:9001"
DEFAULT_JERSEY_PROMPT = (
    "Read the basketball jersey number in this crop. Return only the number from 0 to 99, "
    "including 00 when visible. If the number is not readable, return UNKNOWN."
)


class RoboflowJerseyAnalyzer(TeamOnlyAnalyzer):
    """Track-aware jersey OCR using Roboflow's basketball jersey multimodal model.

    CourtFlux sends a conservative central torso crop to the fine-tuned multimodal model,
    keeps several recent observations per track, and selects the sharpest recent crop for
    each OCR attempt. Identity locking remains a downstream temporal decision; one VLM
    response is never treated as stable identity.

    The endpoint is configurable because multimodal deployments may run on a self-hosted
    Roboflow Inference server, a Dedicated Deployment, or another compatible endpoint.
    API keys are read only from the environment.
    """

    identity_confirmations_required = 3
    preferred_interval_frames = 5

    def __init__(
        self,
        profiling_path: str,
        *,
        api_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        crop_buffer_size: int = 5,
    ) -> None:
        super().__init__(profiling_path=profiling_path)
        self.api_url = str(
            api_url or os.getenv("ROBOFLOW_JERSEY_API_URL") or DEFAULT_JERSEY_API_URL
        ).rstrip("/")
        self.model_id = str(
            model_id or os.getenv("ROBOFLOW_JERSEY_MODEL_ID") or DEFAULT_JERSEY_MODEL_ID
        )
        self.api_key = api_key if api_key is not None else os.getenv("ROBOFLOW_API_KEY")
        self.prompt = str(
            prompt or os.getenv("ROBOFLOW_JERSEY_PROMPT") or DEFAULT_JERSEY_PROMPT
        )
        self.crop_buffer_size = max(1, min(int(crop_buffer_size), 12))
        self.track_crop_buffers: dict[int, list[np.ndarray]] = {}

        try:
            from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        except ImportError as exc:
            raise RuntimeError(
                "Roboflow jersey OCR requires inference-sdk. Install requirements-roboflow.txt."
            ) from exc

        client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
        try:
            client = client.configure(InferenceConfiguration(api_key_transport="header"))
        except Exception:
            # Older/local inference servers may not expose this configuration field.
            pass
        if not hasattr(client, "infer_lmm"):
            raise RuntimeError(
                "Installed inference-sdk does not expose infer_lmm(); upgrade requirements-roboflow.txt."
            )
        self.client = client

    @staticmethod
    def _torso_crop(crop: np.ndarray) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        height, width = crop.shape[:2]
        if height < 36 or width < 20:
            return None
        y1, y2 = int(height * 0.08), int(height * 0.72)
        x1, x2 = int(width * 0.10), int(width * 0.90)
        torso = crop[y1:y2, x1:x2]
        return torso if torso.size else None

    @staticmethod
    def _crop_quality(crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return -1.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area = float(crop.shape[0] * crop.shape[1])
        return sharpness * max(area ** 0.5, 1.0)

    def observe_track(self, crop: np.ndarray, track_id: int) -> None:
        if int(track_id) < 0:
            return
        torso = self._torso_crop(crop)
        if torso is None:
            return
        buffer = self.track_crop_buffers.setdefault(int(track_id), [])
        buffer.append(torso.copy())
        if len(buffer) > self.crop_buffer_size:
            del buffer[:-self.crop_buffer_size]
        if len(self.track_crop_buffers) > 250:
            keep_ids = list(self.track_crop_buffers)[-120:]
            self.track_crop_buffers = {
                key: self.track_crop_buffers[key] for key in keep_ids
            }

    def ready(self, track_id: int) -> bool:
        return bool(self.track_crop_buffers.get(int(track_id)))

    @classmethod
    def _collect_text(cls, value: Any) -> list[str]:
        """Read explicit output fields without interpreting metadata strings as OCR."""
        texts: list[str] = []
        if isinstance(value, str):
            if value.strip():
                texts.append(value.strip())
            return texts
        if isinstance(value, dict):
            preferred = (
                "text",
                "output",
                "result",
                "response",
                "prediction",
                "predictions",
                "answer",
                "generated_text",
            )
            for key in preferred:
                if key in value:
                    texts.extend(cls._collect_text(value[key]))
            if texts:
                return texts
            # Some SDK responses wrap output in unnamed nested containers. Traverse only
            # structured children so metadata like model_id='.../7' cannot become OCR 7.
            for item in value.values():
                if isinstance(item, (dict, list, tuple)):
                    texts.extend(cls._collect_text(item))
            return texts
        if isinstance(value, (list, tuple)):
            for item in value:
                texts.extend(cls._collect_text(item))
        return texts

    @classmethod
    def parse_number(cls, response: Any) -> tuple[str | None, str]:
        """Parse one unambiguous 0-99 jersey number from heterogeneous VLM responses."""
        texts = cls._collect_text(response)
        if not texts:
            return None, "no_text_output"

        candidates: set[str] = set()
        saw_long_numeric = False
        for text in texts:
            if re.search(r"(?<!\d)\d{3,}(?!\d)", text):
                saw_long_numeric = True
            for token in re.findall(r"(?<!\d)\d{1,2}(?!\d)", text):
                value = int(token)
                if 0 <= value <= 99:
                    canonical = "00" if token == "00" else str(value)
                    candidates.add(canonical)

        if len(candidates) == 1 and not saw_long_numeric:
            return next(iter(candidates)), "parsed"
        if len(candidates) > 1:
            return None, "ambiguous_numeric_output"
        if saw_long_numeric:
            return None, "out_of_range_or_malformed"
        return None, "no_valid_number"

    def _valid_roster_numbers(self, team_name: str | None) -> set[str]:
        if not team_name:
            return set()
        team = self.rosters.get(str(team_name), {}) or {}
        roster = team.get("roster", {}) or {}
        valid: set[str] = set()
        for raw in roster:
            text = str(raw).strip()
            if text == "00":
                valid.add("00")
            elif text.isdigit() and 0 <= int(text) <= 99:
                valid.add(str(int(text)))
        return valid

    def predict_jersey(self, track_id: int, team_name: str | None = None) -> dict[str, Any]:
        track_id = int(track_id)
        crops = self.track_crop_buffers.get(track_id, [])
        if not crops:
            return {"accepted": False, "rejection_reason": "crop_not_ready"}

        crop = max(crops, key=self._crop_quality)
        response = self.client.infer_lmm(
            inference_input=crop,
            model_id=self.model_id,
            prompt=self.prompt,
            max_new_tokens=16,
        )
        number, parse_reason = self.parse_number(response)

        if number is None:
            accepted = False
            rejection_reason = parse_reason
        else:
            roster_numbers = self._valid_roster_numbers(team_name)
            if roster_numbers and number not in roster_numbers:
                accepted = False
                rejection_reason = "number_not_in_team_roster"
            else:
                accepted = True
                rejection_reason = "accepted"

        return {
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "candidate_number": number,
            "candidate_posterior": 1.0 if accepted else 0.0,
            "candidate_margin": 1.0 if accepted else 0.0,
            "team": team_name,
            "model_id": self.model_id,
            "backend": "roboflow_multimodal_lmm",
            "api_url": self.api_url,
            "crop_buffer_size": len(crops),
            "parse_reason": parse_reason,
        }

    def get_shirt_number(
        self,
        crop: np.ndarray,
        track_id: int = 0,
        team_name: str | None = None,
    ) -> str:
        self.observe_track(crop, track_id)
        result = self.predict_jersey(track_id, team_name)
        return str(result.get("candidate_number") or "") if result.get("accepted") else ""
