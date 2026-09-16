from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.research_jersey_region_analyzer import (
    ResearchNumberRegionPARSeqAnalyzer,
)


def test_component_candidates_find_digit_like_shapes() -> None:
    mask = np.zeros((100, 80), dtype=np.uint8)
    cv2.rectangle(mask, (28, 28), (36, 62), 255, -1)
    cv2.rectangle(mask, (42, 28), (51, 62), 255, -1)
    boxes = ResearchNumberRegionPARSeqAnalyzer._component_candidates(mask)
    assert len(boxes) >= 2


def test_group_boxes_builds_two_digit_region() -> None:
    boxes = [(28, 28, 9, 35), (42, 28, 10, 35)]
    groups = ResearchNumberRegionPARSeqAnalyzer._group_boxes(boxes)
    assert any(group[4] == 2 for group in groups)


def test_iou_rejects_duplicate_region() -> None:
    overlap = ResearchNumberRegionPARSeqAnalyzer._iou(
        (10, 10, 40, 50),
        (12, 12, 42, 52),
    )
    separate = ResearchNumberRegionPARSeqAnalyzer._iou(
        (10, 10, 30, 30),
        (50, 50, 70, 70),
    )
    assert overlap > 0.68
    assert separate == 0.0


def main() -> None:
    test_component_candidates_find_digit_like_shapes()
    test_group_boxes_builds_two_digit_region()
    test_iou_rejects_duplicate_region()
    print("jersey_region_proposal_test: PASS")


if __name__ == "__main__":
    main()
