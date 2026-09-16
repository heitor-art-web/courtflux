from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.research_jersey_legibility_analyzer import (  # noqa: E402
    ResearchLegibilityRegionPARSeqAnalyzer,
)


def test_preprocess_matches_resnet_contract() -> None:
    crop = np.full((96, 48, 3), 127, dtype=np.uint8)
    tensor = ResearchLegibilityRegionPARSeqAnalyzer._prepare_legibility_crop(crop)
    assert tuple(tensor.shape) == (3, 256, 256)
    assert tensor.dtype == torch.float32
    assert bool(torch.isfinite(tensor).all())


def test_gate_rejects_low_scores_and_ranks_top_k() -> None:
    items = [
        {"ordinal": 1, "frame_id": 10, "quality": 0.95},
        {"ordinal": 2, "frame_id": 11, "quality": 0.80},
        {"ordinal": 3, "frame_id": 12, "quality": 0.70},
    ]
    selected = ResearchLegibilityRegionPARSeqAnalyzer._gate_ranked_items(
        items,
        [0.42, 0.91, 0.72],
        threshold=0.50,
        top_k=2,
    )
    assert [item["frame_id"] for item in selected] == [11, 12]
    assert all(float(item["legibility_model_score"]) >= 0.50 for item in selected)


def test_gate_can_abstain_completely() -> None:
    selected = ResearchLegibilityRegionPARSeqAnalyzer._gate_ranked_items(
        [{"ordinal": 1, "frame_id": 10, "quality": 1.0}],
        [0.12],
        threshold=0.50,
        top_k=5,
    )
    assert selected == []


def main() -> None:
    test_preprocess_matches_resnet_contract()
    test_gate_rejects_low_scores_and_ranks_top_k()
    test_gate_can_abstain_completely()
    print("jersey_legibility_gate_test: PASS")


if __name__ == "__main__":
    main()
