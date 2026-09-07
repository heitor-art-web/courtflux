from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SmartBallPipelineConfig:
    enabled: bool = True
    verify_enabled: bool = True
    zoom_enabled: bool = True
    crop_size: int = 640
    verify_trigger_conf: float = 0.60
    verify_max_candidates: int = 1
    verify_match_px: float = 80.0
    zoom_trigger_rank: float = 0.55
    temporal_zoom_max_age: int = 5
    dedupe_center_px: float = 28.0
    detector_weight: float = 0.60
    verifier_weight: float = 0.20
    temporal_weight: float = 0.15
    geometry_weight: float = 0.05


class SmartBallCandidatePipeline:
    """Conservative detect -> verify -> selectively zoom -> rank pipeline.

    The full-frame detector candidates are never removed. Verification and zoom use the
    same detector checkpoint at the same confidence/imgsz operating point. The pipeline
    can therefore add evidence and candidate proposals without silently changing the
    frozen full-frame detector threshold.

    Temporal state is used only to propose a search ROI and to contribute a soft ranking
    prior. It never creates a ball observation by itself.
    """

    def __init__(
        self,
        *,
        model: Any,
        ball_class_ids: list[int],
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        device: str,
        config: SmartBallPipelineConfig,
    ) -> None:
        self.model = model
        self.ball_class_ids = list(ball_class_ids)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.device = str(device)
        self.config = config

        weights = [
            max(0.0, float(config.detector_weight)),
            max(0.0, float(config.verifier_weight)),
            max(0.0, float(config.temporal_weight)),
            max(0.0, float(config.geometry_weight)),
        ]
        total = sum(weights)
        if total <= 0.0:
            weights = [1.0, 0.0, 0.0, 0.0]
            total = 1.0
        self.weights = [value / total for value in weights]

    @staticmethod
    def crop_box_around_center(
        width: int,
        height: int,
        center: tuple[float, float] | list[float],
        crop_size: int,
    ) -> tuple[int, int, int, int]:
        crop_w = min(max(32, int(crop_size)), max(int(width), 1))
        crop_h = min(max(32, int(crop_size)), max(int(height), 1))
        cx, cy = float(center[0]), float(center[1])
        x1 = int(round(cx - crop_w / 2.0))
        y1 = int(round(cy - crop_h / 2.0))
        x1 = max(0, min(x1, max(int(width) - crop_w, 0)))
        y1 = max(0, min(y1, max(int(height) - crop_h, 0)))
        return x1, y1, x1 + crop_w, y1 + crop_h

    @staticmethod
    def _candidate_center(candidate: dict[str, Any]) -> tuple[float, float] | None:
        center = candidate.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            return float(center[0]), float(center[1])
        box = candidate.get("box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            return (
                (float(box[0]) + float(box[2])) / 2.0,
                (float(box[1]) + float(box[3])) / 2.0,
            )
        return None

    @staticmethod
    def _box_iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in a]
        bx1, by1, bx2, by2 = [float(value) for value in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _geometry_score(candidate: dict[str, Any], frame_shape: tuple[int, ...]) -> float:
        box = candidate.get("box") or []
        if len(box) != 4:
            return 0.5
        width = max(float(box[2]) - float(box[0]), 1.0)
        height = max(float(box[3]) - float(box[1]), 1.0)
        ratio = max(width / height, 1e-6)
        aspect_score = exp(-abs(log(ratio)))

        frame_h, frame_w = frame_shape[:2]
        frame_min = max(float(min(frame_w, frame_h)), 1.0)
        relative_max_side = max(width, height) / frame_min
        size_score = 1.0 if relative_max_side <= 0.08 else max(0.0, 1.0 - (relative_max_side - 0.08) / 0.22)
        return float(0.65 * aspect_score + 0.35 * size_score)

    @staticmethod
    def _temporal_prediction(
        tracker: Any,
        frame_id: int,
    ) -> tuple[tuple[float, float] | None, int | None]:
        last_center = getattr(tracker, "last_center", None)
        last_frame_id = getattr(tracker, "last_frame_id", None)
        if last_center is None or last_frame_id is None:
            return None, None
        age = max(int(frame_id) - int(last_frame_id), 0)
        velocity = getattr(tracker, "velocity", None)
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float64)
        predicted = np.asarray(last_center, dtype=np.float64) + np.asarray(velocity, dtype=np.float64) * float(age)
        return (float(predicted[0]), float(predicted[1])), int(age)

    @staticmethod
    def _temporal_score(
        candidate: dict[str, Any],
        *,
        tracker: Any,
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> tuple[float, bool | None]:
        inspect = getattr(tracker, "inspect_candidate", None)
        if inspect is None:
            return 0.5, None
        details = inspect(candidate, frame_id=frame_id, frame_shape=frame_shape)
        if details is None:
            return 0.5, None
        proximity = max(0.0, min(float(getattr(details, "proximity", 0.0)), 1.0))
        plausible = bool(getattr(details, "plausible", False))
        return proximity, plausible

    def _rank_candidate(
        self,
        candidate: dict[str, Any],
        *,
        tracker: Any,
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        ranked = dict(candidate)
        detector_conf = max(0.0, min(float(ranked.get("detector_confidence", ranked.get("confidence", 0.0))), 1.0))
        ranked["detector_confidence"] = detector_conf

        verification = ranked.get("verification_score")
        verifier_score = detector_conf if verification is None else max(0.0, min(float(verification), 1.0))
        temporal_score, plausible = self._temporal_score(
            ranked,
            tracker=tracker,
            frame_id=frame_id,
            frame_shape=frame_shape,
        )
        geometry_score = self._geometry_score(ranked, frame_shape)
        w_detector, w_verifier, w_temporal, w_geometry = self.weights
        rank_score = (
            w_detector * detector_conf
            + w_verifier * verifier_score
            + w_temporal * temporal_score
            + w_geometry * geometry_score
        )
        ranked["rank_score"] = float(max(0.0, min(rank_score, 1.0)))
        ranked["temporal_score"] = float(temporal_score)
        ranked["temporal_plausible"] = plausible
        ranked["geometry_score"] = float(geometry_score)
        return ranked

    def _predict_crop(
        self,
        frame: Any,
        *,
        crop_box: tuple[int, int, int, int],
        source: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2]
        if getattr(crop, "size", 0) == 0:
            return []
        results = self.model.predict(
            crop,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=self.ball_class_ids,
            device=self.device,
            verbose=False,
        )
        candidates: list[dict[str, Any]] = []
        if not results or getattr(results[0], "boxes", None) is None:
            return candidates
        for box in results[0].boxes:
            conf = float(box.conf[0])
            bx1, by1, bx2, by2 = [float(value) for value in box.xyxy[0].tolist()]
            global_box = [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]
            if global_box[2] <= global_box[0] or global_box[3] <= global_box[1]:
                continue
            center = (
                (global_box[0] + global_box[2]) / 2.0,
                (global_box[1] + global_box[3]) / 2.0,
            )
            candidates.append(
                {
                    "box": [int(round(value)) for value in global_box],
                    "center": center,
                    "confidence": conf,
                    "detector_confidence": conf,
                    "verification_score": conf,
                    "source": source,
                    "zoom_reason": reason,
                    "crop_box": [int(value) for value in crop_box],
                    "interpolated": False,
                }
            )
        return candidates

    def _verify_candidates(
        self,
        frame: Any,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        if not self.config.verify_enabled or not candidates:
            return candidates, [], 0

        ordered = sorted(
            enumerate(candidates),
            key=lambda item: float(item[1].get("detector_confidence", item[1].get("confidence", 0.0))),
            reverse=True,
        )
        top_conf = float(ordered[0][1].get("detector_confidence", ordered[0][1].get("confidence", 0.0)))
        ambiguous = len(candidates) > 1 or top_conf < float(self.config.verify_trigger_conf)
        if not ambiguous:
            return candidates, [], 0

        frame_h, frame_w = frame.shape[:2]
        verified = [dict(candidate) for candidate in candidates]
        refinements: list[dict[str, Any]] = []
        attempts = 0
        for index, seed in ordered[: max(0, int(self.config.verify_max_candidates))]:
            center = self._candidate_center(seed)
            if center is None:
                continue
            attempts += 1
            crop_box = self.crop_box_around_center(
                frame_w,
                frame_h,
                center,
                int(self.config.crop_size),
            )
            crop_candidates = self._predict_crop(
                frame,
                crop_box=crop_box,
                source="zoom_candidate_refine",
                reason="candidate_verification",
            )
            if not crop_candidates:
                verified[index]["verification_score"] = 0.0
                verified[index]["verified"] = False
                continue

            nearest = min(
                crop_candidates,
                key=lambda candidate: float(
                    np.linalg.norm(
                        np.asarray(self._candidate_center(candidate), dtype=np.float64)
                        - np.asarray(center, dtype=np.float64)
                    )
                ),
            )
            nearest_center = self._candidate_center(nearest)
            distance = float(
                np.linalg.norm(
                    np.asarray(nearest_center, dtype=np.float64)
                    - np.asarray(center, dtype=np.float64)
                )
            )
            seed_box = seed.get("box") or [0, 0, 0, 0]
            seed_scale = max(
                float(seed_box[2]) - float(seed_box[0]),
                float(seed_box[3]) - float(seed_box[1]),
                1.0,
            )
            match_radius = max(float(self.config.verify_match_px), 1.5 * seed_scale)
            matched = distance <= match_radius
            verified[index]["verification_score"] = (
                float(nearest.get("detector_confidence", nearest.get("confidence", 0.0)))
                if matched
                else 0.0
            )
            verified[index]["verified"] = bool(matched)
            verified[index]["verification_distance_px"] = round(distance, 4)
            if matched:
                refined = dict(nearest)
                refined["parent_source"] = str(seed.get("source") or "full_frame")
                refined["parent_detector_confidence"] = float(
                    seed.get("detector_confidence", seed.get("confidence", 0.0))
                )
                refinements.append(refined)
        return verified, refinements, attempts

    def _should_temporal_zoom(
        self,
        candidates: list[dict[str, Any]],
        *,
        tracker: Any,
        frame_id: int,
    ) -> tuple[bool, str | None, tuple[float, float] | None]:
        if not self.config.zoom_enabled:
            return False, None, None
        predicted, age = self._temporal_prediction(tracker, frame_id)
        if predicted is None or age is None or age <= 0 or age > int(self.config.temporal_zoom_max_age):
            return False, None, predicted
        if not candidates:
            return True, "no_full_frame_candidate", predicted
        plausible = [candidate for candidate in candidates if candidate.get("temporal_plausible") is True]
        if not plausible:
            return True, "no_temporally_plausible_candidate", predicted
        best_rank = max(float(candidate.get("rank_score", 0.0)) for candidate in plausible)
        if best_rank < float(self.config.zoom_trigger_rank):
            return True, "weak_temporally_plausible_candidate", predicted
        return False, None, predicted

    def _dedupe(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(candidates, key=lambda candidate: float(candidate.get("rank_score", 0.0)), reverse=True)
        kept: list[dict[str, Any]] = []
        for candidate in ordered:
            center = self._candidate_center(candidate)
            box = candidate.get("box") or []
            duplicate_index = None
            for index, existing in enumerate(kept):
                existing_center = self._candidate_center(existing)
                existing_box = existing.get("box") or []
                center_close = False
                if center is not None and existing_center is not None:
                    center_close = float(
                        np.linalg.norm(
                            np.asarray(center, dtype=np.float64)
                            - np.asarray(existing_center, dtype=np.float64)
                        )
                    ) <= float(self.config.dedupe_center_px)
                overlap = (
                    len(box) == 4
                    and len(existing_box) == 4
                    and self._box_iou(list(box), list(existing_box)) >= 0.50
                )
                if center_close or overlap:
                    duplicate_index = index
                    break
            if duplicate_index is None:
                item = dict(candidate)
                item["merged_sources"] = [str(item.get("source") or "unknown")]
                kept.append(item)
                continue
            existing = kept[duplicate_index]
            sources = list(existing.get("merged_sources") or [str(existing.get("source") or "unknown")])
            source = str(candidate.get("source") or "unknown")
            if source not in sources:
                sources.append(source)
            existing["merged_sources"] = sources
        return kept

    def enhance(
        self,
        frame: Any,
        candidates: list[dict[str, Any]],
        *,
        frame_id: int,
        tracker: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = perf_counter()
        if not self.config.enabled:
            return list(candidates), {
                "enabled": False,
                "raw_candidate_count": len(candidates),
                "output_candidate_count": len(candidates),
                "verify_attempts": 0,
                "zoom_triggered": False,
                "zoom_reason": None,
                "zoom_candidate_count": 0,
                "timing_ms": 0.0,
            }

        normalized: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            item.setdefault("source", "full_frame")
            item.setdefault("detector_confidence", float(item.get("confidence", 0.0)))
            item.setdefault("verification_score", None)
            normalized.append(item)

        verified, refinements, verify_attempts = self._verify_candidates(frame, normalized)
        ranked = [
            self._rank_candidate(
                candidate,
                tracker=tracker,
                frame_id=frame_id,
                frame_shape=frame.shape,
            )
            for candidate in verified + refinements
        ]

        zoom_triggered, zoom_reason, predicted = self._should_temporal_zoom(
            ranked,
            tracker=tracker,
            frame_id=frame_id,
        )
        zoom_candidates: list[dict[str, Any]] = []
        zoom_crop_box = None
        if zoom_triggered and predicted is not None:
            frame_h, frame_w = frame.shape[:2]
            zoom_crop_box = self.crop_box_around_center(
                frame_w,
                frame_h,
                predicted,
                int(self.config.crop_size),
            )
            zoom_candidates = self._predict_crop(
                frame,
                crop_box=zoom_crop_box,
                source="zoom_temporal",
                reason=str(zoom_reason),
            )
            ranked.extend(
                self._rank_candidate(
                    candidate,
                    tracker=tracker,
                    frame_id=frame_id,
                    frame_shape=frame.shape,
                )
                for candidate in zoom_candidates
            )

        merged = self._dedupe(ranked)
        merged.sort(key=lambda candidate: float(candidate.get("rank_score", 0.0)), reverse=True)
        elapsed_ms = (perf_counter() - started) * 1000.0
        diagnostics = {
            "enabled": True,
            "raw_candidate_count": len(candidates),
            "verified_candidate_count": sum(1 for candidate in verified if candidate.get("verified") is True),
            "verify_attempts": int(verify_attempts),
            "refinement_candidate_count": len(refinements),
            "zoom_triggered": bool(zoom_triggered),
            "zoom_reason": zoom_reason,
            "zoom_candidate_count": len(zoom_candidates),
            "zoom_crop_box": list(zoom_crop_box) if zoom_crop_box is not None else None,
            "output_candidate_count": len(merged),
            "sources": {
                source: sum(1 for candidate in merged if str(candidate.get("source")) == source)
                for source in sorted({str(candidate.get("source")) for candidate in merged})
            },
            "timing_ms": round(elapsed_ms, 4),
        }
        return merged, diagnostics
