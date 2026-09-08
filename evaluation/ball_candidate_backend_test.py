from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_candidate_backend import (
    build_hf_candidates,
    normalize_label,
    parse_label_set,
)


def test_label_normalization_is_exact_and_stable() -> None:
    assert normalize_label("Sports_Ball") == "sports ball"
    assert normalize_label(" ball-in-basket ") == "ball in basket"
    labels = parse_label_set(None)
    assert labels == {"sports ball", "ball", "basketball"}
    assert "ball in basket" not in labels


def test_hf_candidate_conversion_filters_non_ball_labels() -> None:
    candidates = build_hf_candidates(
        boxes=[
            [10.2, 20.1, 30.4, 40.7],
            [100.0, 100.0, 140.0, 180.0],
        ],
        scores=[0.81, 0.99],
        labels=[1, 2],
        id2label={1: "sports ball", 2: "person"},
        allowed_labels={"sports ball"},
        max_det=100,
        model_id="example/model",
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["box"] == [10, 20, 30, 41]
    assert candidate["center"] == (20.0, 30.5)
    assert candidate["confidence"] == 0.81
    assert candidate["detector_backend"] == "huggingface"
    assert candidate["detector_model"] == "example/model"
    assert candidate["detector_label"] == "sports ball"
    assert candidate["source"] == "full_frame"


def test_hf_candidate_conversion_applies_max_det_after_ranking() -> None:
    candidates = build_hf_candidates(
        boxes=[
            [0, 0, 10, 10],
            [20, 20, 30, 30],
            [40, 40, 50, 50],
        ],
        scores=[0.20, 0.90, 0.50],
        labels=[1, 1, 1],
        id2label={1: "ball"},
        allowed_labels={"ball"},
        max_det=2,
        model_id="example/model",
    )
    assert [round(item["confidence"], 2) for item in candidates] == [0.90, 0.50]


def test_invalid_boxes_are_dropped() -> None:
    candidates = build_hf_candidates(
        boxes=[[10, 10, 10, 20], [30, 30, 20, 40]],
        scores=[0.9, 0.8],
        labels=[1, 1],
        id2label={1: "ball"},
        allowed_labels={"ball"},
        max_det=10,
        model_id="example/model",
    )
    assert candidates == []


def main() -> None:
    test_label_normalization_is_exact_and_stable()
    test_hf_candidate_conversion_filters_non_ball_labels()
    test_hf_candidate_conversion_applies_max_det_after_ranking()
    test_invalid_boxes_are_dropped()
    print("[PASS] ball candidate backend")


if __name__ == "__main__":
    main()
