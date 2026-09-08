from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


# Backward-compatible 14-point map used by the existing deployed Roboflow model.
FIBA_COURT_KEYPOINTS_14: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, 15.0),
    2: (28.0, 0.0),
    3: (28.0, 15.0),
    4: (14.0, 0.0),
    5: (14.0, 15.0),
    6: (0.0, 5.05),
    7: (0.0, 9.95),
    8: (5.8, 5.05),
    9: (5.8, 9.95),
    10: (22.2, 5.05),
    11: (22.2, 9.95),
    12: (28.0, 5.05),
    13: (28.0, 9.95),
}


@dataclass(frozen=True)
class HomographyEstimate:
    matrix: np.ndarray | None
    anchor_count: int
    inlier_count: int
    reprojection_rmse_px: float | None
    valid: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "anchor_count": int(self.anchor_count),
            "inlier_count": int(self.inlier_count),
            "reprojection_rmse_px": (
                None if self.reprojection_rmse_px is None else round(float(self.reprojection_rmse_px), 4)
            ),
        }


def load_keypoint_map(path: str | Path | None) -> dict[int, tuple[float, float]]:
    """Load a versioned image-keypoint-index -> FIBA court-meter map.

    Accepted JSON forms:
      {"0": [0.0, 0.0], "1": [0.0, 15.0], ...}
    or
      {"points": [{"index": 0, "court_xy_m": [0.0, 0.0]}, ...]}

    If no path is supplied, the frozen 14-point map is returned. CourtFlux deliberately
    does not guess the ordering of Roboflow's 33-point schema; export/version that map
    explicitly before using a 33-keypoint checkpoint for metric homography.
    """
    if path is None or not str(path).strip():
        return dict(FIBA_COURT_KEYPOINTS_14)

    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    result: dict[int, tuple[float, float]] = {}
    if isinstance(payload, Mapping) and "points" in payload:
        for item in payload["points"]:
            idx = int(item["index"])
            xy = item["court_xy_m"]
            result[idx] = (float(xy[0]), float(xy[1]))
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            result[int(key)] = (float(value[0]), float(value[1]))
    else:
        raise ValueError("Court keypoint map must be a JSON object.")

    if len(result) < 4:
        raise ValueError("Court keypoint map requires at least four indexed court points.")
    return result


def estimate_homography(
    src_points_px: Sequence[Sequence[float]],
    dst_points_m: Sequence[Sequence[float]],
    *,
    ransac_reprojection_px: float = 5.0,
    max_rmse_px: float = 12.0,
) -> HomographyEstimate:
    if len(src_points_px) < 4 or len(dst_points_m) != len(src_points_px):
        return HomographyEstimate(None, len(src_points_px), 0, None, False)

    src = np.asarray(src_points_px, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_points_m, dtype=np.float32).reshape(-1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, float(ransac_reprojection_px))
    if H is None or not np.isfinite(H).all():
        return HomographyEstimate(None, len(src), 0, None, False)

    inlier_mask = np.ones(len(src), dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 4:
        return HomographyEstimate(H, len(src), inlier_count, None, False)

    try:
        H_inv = np.linalg.inv(H)
        dst_h = cv2.convertPointsToHomogeneous(dst.reshape(-1, 1, 2)).reshape(-1, 3).T
        projected = (H_inv @ dst_h).T
        projected = projected[:, :2] / projected[:, 2:3]
        errors = np.linalg.norm(projected[inlier_mask] - src[inlier_mask], axis=1)
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else None
    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
        rmse = None

    valid = bool(inlier_count >= 4 and rmse is not None and np.isfinite(rmse) and rmse <= max_rmse_px)
    return HomographyEstimate(H, len(src), inlier_count, rmse, valid)


def project_image_point_to_court(
    point_xy: Sequence[float],
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    if homography is None or len(point_xy) < 2:
        return None
    point = np.asarray([float(point_xy[0]), float(point_xy[1]), 1.0], dtype=np.float64)
    projected = np.asarray(homography, dtype=np.float64) @ point
    if abs(float(projected[2])) < 1e-9:
        return None
    xy = projected[:2] / projected[2]
    if not np.isfinite(xy).all():
        return None
    return float(xy[0]), float(xy[1])


def annotate_projected_entities(frame_state: Any) -> None:
    """Attach planar court projections without changing detector/tracker decisions.

    Player bottom-centers are suitable ground-plane points. Ball center projection is
    explicitly labelled planar/apparent because airborne ball height violates the plane
    assumption of a single court homography.
    """
    H = getattr(frame_state, "homography_matrix", None)
    if H is None:
        return

    for player in getattr(frame_state, "players", []) or []:
        box = player.get("box") or []
        if len(box) == 4:
            foot = ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))
            court_xy = project_image_point_to_court(foot, H)
            if court_xy is not None:
                player["court_xy"] = [round(court_xy[0], 4), round(court_xy[1], 4)]
                player["court_projection_source"] = "homography_ground_plane"

    ball = getattr(frame_state, "ball", None)
    if isinstance(ball, dict):
        center = ball.get("center")
        if center is not None:
            court_xy = project_image_point_to_court(center, H)
            if court_xy is not None:
                ball["projected_court_xy"] = [round(court_xy[0], 4), round(court_xy[1], 4)]
                ball["court_projection_source"] = "homography_planar_apparent"
