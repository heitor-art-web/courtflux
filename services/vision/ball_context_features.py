from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from services.vision.court_geometry import project_image_point_to_court


FIBA_LENGTH_M = 28.0
FIBA_WIDTH_M = 15.0
FIBA_HOOPS_M = ((1.575, 7.5), (26.425, 7.5))


def _point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _distance_point_to_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return float(math.hypot(dx, dy))


def _court_margin(xy: tuple[float, float]) -> float:
    x, y = xy
    return float(min(x, FIBA_LENGTH_M - x, y, FIBA_WIDTH_M - y))


def _nearest_hoop_distance(xy: tuple[float, float]) -> float:
    point = np.asarray(xy, dtype=np.float64)
    return float(min(np.linalg.norm(point - np.asarray(hoop, dtype=np.float64)) for hoop in FIBA_HOOPS_M))


@dataclass
class _CourtHistory:
    frame_id: int
    xy: np.ndarray


class BallContextFeatureEnricher:
    """Attach court/player context to OC-SORT hypotheses without changing decisions.

    Ball court coordinates are explicitly *planar apparent* projections. A basketball can
    be airborne, so homography-derived ball position is never treated as ground truth or a
    hard validity rule. This class only emits diagnostic features for later offline
    separability / arbitration analysis.
    """

    def __init__(self) -> None:
        self._court_history: dict[str, _CourtHistory] = {}

    def _player_context(
        self,
        center_px: tuple[float, float],
        court_xy: tuple[float, float] | None,
        players: list[dict[str, Any]],
    ) -> dict[str, Any]:
        nearest_image_norm: float | None = None
        nearest_image_track_id: int | None = None
        inside_player_box = False

        nearest_court_m: float | None = None
        nearest_court_track_id: int | None = None
        if court_xy is not None:
            court_point = np.asarray(court_xy, dtype=np.float64)
        else:
            court_point = None

        for player in players:
            box = _box(player.get("box"))
            if box is not None:
                x1, y1, x2, y2 = box
                height = max(y2 - y1, 1.0)
                distance_norm = _distance_point_to_box(center_px, box) / height
                if nearest_image_norm is None or distance_norm < nearest_image_norm:
                    nearest_image_norm = float(distance_norm)
                    try:
                        nearest_image_track_id = int(player.get("track_id"))
                    except (TypeError, ValueError):
                        nearest_image_track_id = None
                if x1 <= center_px[0] <= x2 and y1 <= center_px[1] <= y2:
                    inside_player_box = True

            player_court = _point(player.get("court_xy"))
            if court_point is not None and player_court is not None:
                distance_m = float(
                    np.linalg.norm(court_point - np.asarray(player_court, dtype=np.float64))
                )
                if nearest_court_m is None or distance_m < nearest_court_m:
                    nearest_court_m = distance_m
                    try:
                        nearest_court_track_id = int(player.get("track_id"))
                    except (TypeError, ValueError):
                        nearest_court_track_id = None

        return {
            "player_count": int(len(players)),
            "nearest_player_box_distance_norm": (
                None if nearest_image_norm is None else round(nearest_image_norm, 6)
            ),
            "nearest_player_box_track_id": nearest_image_track_id,
            "inside_player_box": bool(inside_player_box),
            "nearest_player_court_distance_m": (
                None if nearest_court_m is None else round(nearest_court_m, 6)
            ),
            "nearest_player_court_track_id": nearest_court_track_id,
        }

    def enrich(self, frame_state: Any) -> dict[str, Any]:
        detector = frame_state.diagnostics.get("detector", {}) or {}
        association = detector.get("ball_association", {}) or {}
        ball_hypotheses = association.get("ball_hypotheses", {}) or {}
        hypotheses = ball_hypotheses.get("hypotheses") or []
        if not isinstance(hypotheses, list) or not hypotheses:
            return {"enabled": False, "reason": "no_ball_hypotheses", "enriched": 0}

        camera = frame_state.diagnostics.get("camera", {}) or {}
        homography_valid = bool(camera.get("valid", False))
        H = getattr(frame_state, "homography_matrix", None) if homography_valid else None
        players = [p for p in (getattr(frame_state, "players", []) or []) if isinstance(p, dict)]
        frame_id = int(getattr(frame_state, "frame_id", 0))

        rmse = camera.get("reprojection_rmse_px")
        try:
            rmse_value = None if rmse is None else float(rmse)
        except (TypeError, ValueError):
            rmse_value = None
        anchors = int(camera.get("anchor_count", 0) or 0)
        inliers = int(camera.get("inlier_count", 0) or 0)
        inlier_ratio = float(inliers / anchors) if anchors > 0 else None

        enriched = 0
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            center = _point(hypothesis.get("center"))
            if center is None:
                continue

            court_xy = project_image_point_to_court(center, H) if H is not None else None
            apparent_speed = None
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
            if court_xy is not None and hypothesis_id:
                current_xy = np.asarray(court_xy, dtype=np.float64)
                previous = self._court_history.get(hypothesis_id)
                if previous is not None:
                    dt = max(frame_id - int(previous.frame_id), 1)
                    apparent_speed = float(np.linalg.norm(current_xy - previous.xy) / float(dt))
                if bool(hypothesis.get("observed", False)):
                    self._court_history[hypothesis_id] = _CourtHistory(frame_id, current_xy)

            context: dict[str, Any] = {
                "homography_valid": homography_valid,
                "homography_source": camera.get("source"),
                "homography_anchor_count": anchors,
                "homography_inlier_count": inliers,
                "homography_inlier_ratio": (
                    None if inlier_ratio is None else round(inlier_ratio, 6)
                ),
                "homography_rmse_px": (
                    None if rmse_value is None else round(rmse_value, 6)
                ),
                "apparent_court_xy": (
                    None
                    if court_xy is None
                    else [round(float(court_xy[0]), 6), round(float(court_xy[1]), 6)]
                ),
                "court_inside_margin_m": (
                    None if court_xy is None else round(_court_margin(court_xy), 6)
                ),
                "inside_court_1m_margin": bool(
                    court_xy is not None and _court_margin(court_xy) >= -1.0
                ),
                "nearest_hoop_apparent_distance_m": (
                    None
                    if court_xy is None
                    else round(_nearest_hoop_distance(court_xy), 6)
                ),
                "apparent_court_speed_m_per_step": (
                    None if apparent_speed is None else round(apparent_speed, 6)
                ),
                **self._player_context(center, court_xy, players),
            }
            hypothesis["context"] = context
            enriched += 1

        # Bound history to active OC-SORT IDs so long runs do not grow unbounded.
        active_ids = {
            str(item.get("hypothesis_id"))
            for item in hypotheses
            if isinstance(item, dict)
            and item.get("hypothesis_id")
            and item.get("lifecycle") != "REMOVED"
        }
        for key in list(self._court_history):
            if key not in active_ids:
                self._court_history.pop(key, None)

        diagnostics = {
            "enabled": True,
            "decision_effect": "diagnostic_only",
            "enriched": int(enriched),
            "homography_valid": homography_valid,
            "player_count": int(len(players)),
            "ball_projection_semantics": "planar_apparent_not_ground_truth",
        }
        ball_hypotheses["context_features"] = diagnostics
        association["ball_hypotheses"] = ball_hypotheses
        detector["ball_association"] = association
        frame_state.diagnostics["detector"] = detector
        return diagnostics
