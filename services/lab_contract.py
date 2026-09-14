from __future__ import annotations

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
    "jersey_interval_frames": 24,
    "persist_results": True,
    "live_preview": True,
    "preview_interval_frames": 15,
    "single_forward": True,
    "smart_ball_mode": "off",
    "smart_ball_crop_imgsz": 640,
}


def lab_defaults() -> dict:
    return dict(LAB_DEFAULTS)
