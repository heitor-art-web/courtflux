from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATCH_CONFIG_PATH = PROJECT_ROOT / "data" / "match_config.json"
DEFAULT_HF_HOME = PROJECT_ROOT / "weights" / "huggingface"
JERSEY_BACKENDS = (
    "best_temporal",
    "roboflow_classification",
    "smolvlm2_jersey",
    "ebqwen_vl",
    "uncertainty_vit_b",
    "legacy_digit_heads",
    "auto",
)
JERSEY_BACKEND_ALIASES = {
    "temporal": "best_temporal",
    "bilstm": "best_temporal",
    "best_temporal.pt": "best_temporal",
    "roboflow": "roboflow_classification",
    "rf": "roboflow_classification",
    "smolvlm2": "smolvlm2_jersey",
    "smol": "smolvlm2_jersey",
    "ebqwen": "ebqwen_vl",
    "qwen": "ebqwen_vl",
    "vit": "uncertainty_vit_b",
    "vit_b": "uncertainty_vit_b",
    "uncertainty": "uncertainty_vit_b",
    "legacy": "legacy_digit_heads",
    "efficientnet": "legacy_digit_heads",
}

# Keep Hugging Face model artefacts inside the project-local weights tree by default.
# A deliberate HF_HOME supplied by the operator still wins.
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))

LAB_DEFAULTS = {
    "video_source": "https://www.youtube.com/watch?v=OwiQgAMqyUs",
    "model": "weights/model_v1.1.0.pt",
    "detector_conf": 0.40,
    "inference_conf": 0.03,
    "ball_conf": 0.075,
    "imgsz": 960,
    "device": "mps",
    "tracker": "botsort.yaml",
    "detection_only": True,
    "enable_jersey": True,
    "jersey_backend": "best_temporal",
    "jersey_interval_frames": 24,
    # Worker-backed analyses persist platform state/summary/metrics through Supabase.
    # Legacy local SQLite persistence is intentionally disabled for the Lab contract.
    "persist_results": False,
    "live_preview": True,
    "preview_interval_frames": 15,
    "single_forward": True,
    "smart_ball_mode": "off",
    "smart_ball_crop_imgsz": 640,
    "identity_debug": True,
}


def resolve_jersey_backend(value: str | None = None) -> str:
    requested = str(
        value
        if value is not None
        else os.getenv("BASKETBALL_JERSEY_BACKEND", LAB_DEFAULTS["jersey_backend"])
    ).strip().lower()
    requested = JERSEY_BACKEND_ALIASES.get(requested, requested)
    if requested not in JERSEY_BACKENDS:
        raise ValueError(
            "BASKETBALL_JERSEY_BACKEND must be one of: "
            + ", ".join(JERSEY_BACKENDS)
        )
    return requested


def lab_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = dict(LAB_DEFAULTS)
    # Health/UI defaults must reflect the backend selected in the worker process,
    # rather than always advertising best_temporal after an operator export.
    defaults["jersey_backend"] = resolve_jersey_backend()
    defaults["hf_home"] = os.getenv("HF_HOME", str(DEFAULT_HF_HOME))
    defaults["match_config"] = lab_match_config()
    defaults["match_config_source"] = lab_match_config_source()
    return defaults


def lab_match_config() -> dict[str, Any]:
    """Return the repository-canonical match configuration used by Beta Lab."""
    data = json.loads(DEFAULT_MATCH_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Lab match config must be a JSON object: {DEFAULT_MATCH_CONFIG_PATH}")
    return data


def lab_match_config_source() -> str:
    try:
        return str(DEFAULT_MATCH_CONFIG_PATH.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(DEFAULT_MATCH_CONFIG_PATH)
