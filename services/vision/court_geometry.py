from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


# Backward-compatible 14-point map used by the existing deployed court-calibration path.
# This map must not be reused for Roboflow's 33-keypoint schema without an explicit,
# versioned index mapping verified against that dataset/model version.
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
    # Image pixels -> court metres.
    matrix: np.ndarray | None
    anchor_count: int
    inlier_count: int
    # Error is measured after projecting court anchors back to image space.
    reprojection_rmse_px: float | None
    valid: bool

    def diagnostics(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "anchor_count": int(self.anchor_count),
            "inlier_count": int(self.inlier_count),
            "reprojection_rmse_px": (
                None
                if self.reprojection_rmse_px is None
                else round(float(self.reprojection_rmse_px), 4)
            ),
        }


def load_keypoint_map(path: str | Path | None) -> dict[int, tuple[float, float]]:
    """Load a versioned image-keypoint-index -> FIBA court-metre map.

    Accepted JSON forms::

        {"0": [0.0, 0.0], "1": [0.0, 15.0]}

    or::

        {"points": [{"index": 0, "court_xy_m": [0.0, 0.0]}]}

    If no path is supplied, the frozen legacy 14-point map is returned. CourtFlux
    deliberately does not guess the ordering of Roboflow's 33-point schema.
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
    """Estimate an image-pixel -> court-metre homography with pixel-space RANSAC.

    OpenCV applies the RANSAC threshold in the *destination* coordinate system of
    ``findHomography``. Estimating ``src_px -> dst_m`` directly would therefore interpret
    a value such as ``5`` as five metres, not five pixels. To keep the public API and
    diagnostics explicitly pixel based, CourtFlux estimates ``court_m -> image_px`` with
    RANSAC, computes reprojection error in image pixels, and finally inverts the matrix.
    """
    anchor_count = len(src_points_px)
    if anchor_count < 4 or len(dst_points_m) != anchor_count:
        return HomographyEstimate(None, anchor_count, 0, None, False)

    try:
        src = np.asarray(src_points_px, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(dst_points_m, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return HomographyEstimate(None, anchor_count, 0, None, False)

    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        return HomographyEstimate(None, anchor_count, 0, None, False)

    # RANSAC threshold is now genuinely in image pixels because image space is the
    # destination of this estimation.
    court_to_image, mask = cv2.findHomography(
        dst,
        src,
        cv2.RANSAC,
        float(ransac_reprojection_px),
    )
    if court_to_image is None or not np.isfinite(court_to_image).all():
        return HomographyEstimate(None, anchor_count, 0, None, False)

    inlier_mask = (
        np.ones(anchor_count, dtype=bool)
        if mask is None
        else mask.reshape(-1).astype(bool)
    )
    inlier_count = int(inlier_mask.sum())

    try:
        image_to_court = np.linalg.inv(court_to_image)
    except np.linalg.LinAlgError:
        return HomographyEstimate(None, anchor_count, inlier_count, None, False)

    if not np.isfinite(image_to_court).all():
        return HomographyEstimate(None, anchor_count, inlier_count, None, False)
    if inlier_count < 4:
        return HomographyEstimate(image_to_court, anchor_count, inlier_count, None, False)

    try:
        projected = cv2.perspectiveTransform(
            dst.reshape(-1, 1, 2).astype(np.float64),
            court_to_image.astype(np.float64),
        ).reshape(-1, 2)
        errors = np.linalg.norm(projected[inlier_mask] - src[inlier_mask], axis=1)
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else None
    except (cv2.error, FloatingPointError, ValueError):
        rmse = None

    valid = bool(
        inlier_count >= 4
        and rmse is not None
        and np.isfinite(rmse)
        and rmse <= float(max_rmse_px)
    )
    return HomographyEstimate(
        image_to_court,
        anchor_count,
        inlier_count,
        rmse,
        valid,
    )


def project_image_point_to_court(
    point_xy: Sequence[float],
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    if homography is None or len(point_xy) < 2:
        return None
    point = np.asarray(
        [float(point_xy[0]), float(point_xy[1]), 1.0],
        dtype=np.float64,
    )
    projected = np.asarray(homography, dtype=np.float64) @ point
    if abs(float(projected[2])) < 1e-9:
        return None
    xy = projected[:2] / projected[2]
    if not np.isfinite(xy).all():
        return None
    return float(xy[0]), float(xy[1])


def annotate_projected_entities(frame_state: Any) -> None:
    """Attach planar court projections without changing detector/tracker decisions.

    Player bottom-centres are suitable ground-plane points. Ball-centre projection is
    explicitly labelled planar/apparent because airborne ball height violates the plane
    assumption of a single court homography.
    """
    H = getattr(frame_state, "homography_matrix", None)
    if H is None:
        return

    for player in getattr(frame_state, "players", []) or []:
        box = player.get("box") or []
        if len(box) == 4:
            foot = (
                (float(box[0]) + float(box[2])) / 2.0,
                float(box[3]),
            )
            court_xy = project_image_point_to_court(foot, H)
            if court_xy is not None:
                player["court_xy"] = [
                    round(court_xy[0], 4),
                    round(court_xy[1], 4),
                ]
                player["court_projection_source"] = "homography_ground_plane"

    ball = getattr(frame_state, "ball", None)
    if isinstance(ball, dict):
        center = ball.get("center")
        if center is not None:
            court_xy = project_image_point_to_court(center, H)
            if court_xy is not None:
                ball["projected_court_xy"] = [
                    round(court_xy[0], 4),
                    round(court_xy[1], 4),
                ]
                ball["court_projection_source"] = "homography_planar_apparent"
