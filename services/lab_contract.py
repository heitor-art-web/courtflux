from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATCH_CONFIG_PATH = PROJECT_ROOT / "data" / "match_config.json"

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
    "persist_results": True,
    "live_preview": True,
    "preview_interval_frames": 15,
    "single_forward": True,
    "smart_ball_mode": "off",
    "smart_ball_crop_imgsz": 640,
    "identity_debug": True,
}


def lab_defaults() -> dict[str, Any]:
    defaults: dict[str, Any] = dict(LAB_DEFAULTS)
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
