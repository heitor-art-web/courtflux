from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from services.vision.p1_parseq_jersey_analyzer import SoccerNetPARSeqJerseyAnalyzer


class ResearchNumberRegionPARSeqAnalyzer(SoccerNetPARSeqJerseyAnalyzer):
    """Research-only PARSeq adapter with visual number-region proposal scoring.

    The scorer is deliberately roster-blind and OCR-blind. It searches each selected
    torso crop for compact, high-contrast, digit-like connected components and groups
    nearby components into one/two-digit proposals. PARSeq only sees the highest-scoring
    regions. Identity-v2 remains the downstream temporal identity layer.
    """

    REGION_MIN_SCORE = 0.38
    REGION_MAX_PROPOSALS = 3

    def __init__(self, profiling_path: str, device: str = "mps") -> None:
        super().__init__(profiling_path=profiling_path, device=device)
        self.research_roi = True
        self.checkpoint_diagnostics.update(
            {
                "temporal_aggregation": "research_number_region_then_identity_v2",
                "localized_roi_proposals": True,
                "number_region_proposal_scorer": True,
                "number_region_method": "contrast_components_geometry_v1",
                "number_region_min_score": self.REGION_MIN_SCORE,
                "number_region_max_proposals_per_frame": self.REGION_MAX_PROPOSALS,
                "research_roi_names": ["number_region_ranked"],
                "pose_guided_roi": False,
            }
        )

    @staticmethod
    def _component_candidates(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        height, width = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < max(8, int(height * 0.13)) or h > int(height * 0.72):
                continue
            if w < max(3, int(width * 0.025)) or w > int(width * 0.36):
                continue
            aspect = w / max(float(h), 1.0)
            if not 0.10 <= aspect <= 1.15:
                continue
            area_ratio = (w * h) / max(float(width * height), 1.0)
            if not 0.003 <= area_ratio <= 0.18:
                continue
            center_x = (x + 0.5 * w) / max(float(width), 1.0)
            center_y = (y + 0.5 * h) / max(float(height), 1.0)
            if not 0.08 <= center_x <= 0.92 or not 0.02 <= center_y <= 0.90:
                continue
            output.append((x, y, w, h))
        return output

    @staticmethod
    def _group_boxes(
        boxes: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int, int]]:
        groups: list[tuple[int, int, int, int, int]] = []
        for x, y, w, h in boxes:
            groups.append((x, y, x + w, y + h, 1))
        for index, first in enumerate(boxes):
            x1, y1, w1, h1 = first
            cy1 = y1 + 0.5 * h1
            for second in boxes[index + 1 :]:
                x2, y2, w2, h2 = second
                cy2 = y2 + 0.5 * h2
                height_ratio = min(h1, h2) / max(float(max(h1, h2)), 1.0)
                vertical_alignment = abs(cy1 - cy2) / max(float(max(h1, h2)), 1.0)
                gap = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                if height_ratio < 0.58 or vertical_alignment > 0.48:
                    continue
                if gap > 0.95 * max(h1, h2):
                    continue
                groups.append(
                    (
                        min(x1, x2),
                        min(y1, y2),
                        max(x1 + w1, x2 + w2),
                        max(y1 + h1, y2 + h2),
                        2,
                    )
                )
        return groups

    @staticmethod
    def _proposal_score(
        gray: np.ndarray,
        box: tuple[int, int, int, int, int],
    ) -> float:
        height, width = gray.shape[:2]
        x1, y1, x2, y2, count = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return 0.0
        pad_x = max(2, int(round(bw * 0.28)))
        pad_y = max(2, int(round(bh * 0.18)))
        px1, px2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
        py1, py2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
        roi = gray[py1:py2, px1:px2]
        if roi.size == 0:
            return 0.0

        contrast = min(float(np.std(roi)) / 72.0, 1.0)
        edges = cv2.Canny(roi, 55, 145)
        edge_density = float(np.mean(edges > 0))
        edge_score = max(0.0, 1.0 - abs(edge_density - 0.18) / 0.18)
        aspect = bw / max(float(bh), 1.0)
        aspect_score = max(0.0, 1.0 - abs(aspect - (0.72 if count == 2 else 0.42)) / 0.85)
        cx = (x1 + x2) * 0.5 / max(float(width), 1.0)
        cy = (y1 + y2) * 0.5 / max(float(height), 1.0)
        centrality = max(0.0, 1.0 - abs(cx - 0.5) / 0.5)
        vertical_prior = max(0.0, 1.0 - abs(cy - 0.43) / 0.55)
        height_ratio = bh / max(float(height), 1.0)
        size_score = max(0.0, 1.0 - abs(height_ratio - 0.32) / 0.34)
        pair_bonus = 1.0 if count == 2 else 0.72

        return float(
            0.24 * contrast
            + 0.20 * edge_score
            + 0.15 * aspect_score
            + 0.14 * centrality
            + 0.10 * vertical_prior
            + 0.10 * size_score
            + 0.07 * pair_bonus
        )

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / max(float(area_a + area_b - inter), 1.0)

    def _number_region_proposals(self, crop: np.ndarray) -> list[dict[str, Any]]:
        if crop is None or crop.size == 0:
            return []
        height, width = crop.shape[:2]
        if height < 28 or width < 20:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (3, 3), 0)
        _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = cv2.bitwise_not(bright)

        boxes = self._component_candidates(bright) + self._component_candidates(dark)
        unique_boxes = sorted(set(boxes))
        grouped = self._group_boxes(unique_boxes)

        ranked: list[tuple[float, tuple[int, int, int, int, int]]] = []
        for box in grouped:
            score = self._proposal_score(clahe, box)
            if score >= self.REGION_MIN_SCORE:
                ranked.append((score, box))
        ranked.sort(key=lambda item: item[0], reverse=True)

        selected: list[dict[str, Any]] = []
        selected_boxes: list[tuple[int, int, int, int]] = []
        for score, (x1, y1, x2, y2, count) in ranked:
            bw, bh = x2 - x1, y2 - y1
            pad_x = max(3, int(round(bw * 0.34)))
            pad_y = max(3, int(round(bh * 0.22)))
            px1, px2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
            py1, py2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
            candidate_box = (px1, py1, px2, py2)
            if any(self._iou(candidate_box, previous) > 0.68 for previous in selected_boxes):
                continue
            roi = crop[py1:py2, px1:px2]
            if roi.size == 0 or roi.shape[0] < 12 or roi.shape[1] < 12:
                continue
            selected_boxes.append(candidate_box)
            selected.append(
                {
                    "roi_name": f"number_region_{len(selected) + 1}",
                    "roi": roi,
                    "roi_prior": max(0.18, min(1.0, score)),
                    "roi_legibility_score": max(0.0, min(1.0, score)),
                    "region_score": round(score, 6),
                    "component_count": count,
                    "region_box": [px1, py1, px2, py2],
                }
            )
            if len(selected) >= self.REGION_MAX_PROPOSALS:
                break
        return selected

    def _research_roi_proposals(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        crop = item.get("crop")
        if crop is None or crop.size == 0:
            return []
        return self._number_region_proposals(crop)
