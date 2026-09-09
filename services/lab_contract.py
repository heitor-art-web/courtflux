from __future__ import annotations

LAB_DEFAULTS = {
    "video_source": "https://www.youtube.com/watch?v=OwiQgAMqyUs",
    "model": "weights/model_v1.0.0.pt",
    "detector_conf": 0.10,
    "inference_conf": 0.03,
    "ball_conf": 0.03,
    "imgsz": 960,
    "device": "mps",
    "tracker": "botsort.yaml",
    "detection_only": True,
    "enable_jersey": False,
    "jersey_interval_frames": 24,
    "persist_results": True,
    "live_preview": True,
    "preview_interval_frames": 15,
}


def lab_defaults() -> dict:
    return dict(LAB_DEFAULTS)
