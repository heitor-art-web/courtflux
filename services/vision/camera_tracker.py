from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

from services.vision.court_geometry import (
    annotate_projected_entities,
    estimate_homography,
    load_keypoint_map,
    project_image_point_to_court,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

try:
    from inference import get_model as get_local_inference_model

    LOCAL_INFERENCE_AVAILABLE = True
except ImportError:
    get_local_inference_model = None
    LOCAL_INFERENCE_AVAILABLE = False


class _ServerlessCourtKeypointModel:
    """Small compatibility wrapper around the lightweight Roboflow HTTP SDK."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        api_url: str,
        confidence: float,
        keypoint_confidence: float,
    ) -> None:
        try:
            from inference_sdk import InferenceConfiguration, InferenceHTTPClient
        except ImportError as exc:
            raise RuntimeError(
                "Serverless court inference requires optional dependencies. Install: "
                "python3 -m pip install -r requirements-roboflow.txt"
            ) from exc

        self.model_id = str(model_id)
        self.client = InferenceHTTPClient(
            api_url=str(api_url),
            api_key=str(api_key),
        ).configure(
            InferenceConfiguration(
                api_key_transport="header",
                confidence_threshold=float(confidence),
                keypoint_confidence_threshold=float(keypoint_confidence),
                disable_active_learning=True,
            )
        )

    def infer(self, frame: np.ndarray, confidence: float | None = None) -> Any:
        # Confidence is configured on the client to keep the wrapper compatible with
        # the legacy `get_model(...).infer(frame, confidence=...)` contract.
        del confidence
        return self.client.infer(frame, model_id=self.model_id)


class CameraTracker:
    def __init__(
        self,
        first_frame: Optional[np.ndarray] = None,
        initial_H: Optional[np.ndarray] = None,
        initial_polygon: Optional[np.ndarray] = None,
        use_keypoint_model: bool = True,
        model_id: str | None = None,
        confidence: float = 0.3,
        anchor_confidence: float = 0.5,
        keypoint_map_path: str | None = None,
        max_reprojection_rmse_px: float = 12.0,
    ):
        self.frame_h = 0
        self.frame_w = 0
        self.confidence = float(confidence)
        self.anchor_confidence = float(anchor_confidence)
        self.max_reprojection_rmse_px = float(max_reprojection_rmse_px)
        self.model_id = (
            model_id
            or os.getenv("BASKETBALL_COURT_MODEL_ID", "basketball-court-detection-2/14")
        )
        resolved_map_path = keypoint_map_path or os.getenv("BASKETBALL_COURT_KEYPOINT_MAP", "")
        self.keypoint_map = load_keypoint_map(resolved_map_path or None)
        self.keypoint_schema = "external" if resolved_map_path else "fiba14_builtin"
        self.keypoint_model = None
        self.keypoint_backend = os.getenv(
            "BASKETBALL_COURT_INFERENCE_BACKEND",
            "legacy",
        ).strip().lower() or "legacy"
        self.last_calibration_diagnostics: dict[str, Any] = {
            "valid": False,
            "anchor_count": 0,
            "inlier_count": 0,
            "reprojection_rmse_px": None,
            "source": "uninitialized",
            "keypoint_backend": self.keypoint_backend,
        }

        requested_keypoints = bool(use_keypoint_model)
        self.use_keypoint_model = False
        if requested_keypoints:
            self._initialize_keypoint_model()

        self.initial_H = initial_H
        self.base_polygon = initial_polygon
        self.M_accum = np.eye(3, dtype=np.float32)
        self.prev_gray = None
        self.prev_pts = None

        if first_frame is not None:
            self._init_first_frame(first_frame)

    def _initialize_keypoint_model(self) -> None:
        if self.keypoint_backend in {"serverless", "roboflow", "http"}:
            api_key = (
                os.getenv("ROBOFLOW_PRIVATE_API_KEY", "").strip()
                or os.getenv("ROBOFLOW_API_KEY", "").strip()
            )
            if not api_key:
                print(
                    "[CameraTracker] Serverless court inference requested but no private/API key "
                    "is configured; using optical-flow fallback.",
                    flush=True,
                )
                return
            api_url = os.getenv(
                "BASKETBALL_ROBOFLOW_API_URL",
                "https://serverless.roboflow.com",
            ).strip()
            try:
                self.keypoint_model = _ServerlessCourtKeypointModel(
                    model_id=self.model_id,
                    api_key=api_key,
                    api_url=api_url,
                    confidence=self.confidence,
                    keypoint_confidence=self.anchor_confidence,
                )
                self.use_keypoint_model = True
                print(
                    f"[CameraTracker] Loaded Serverless court keypoint model: {self.model_id} "
                    f"schema={self.keypoint_schema} anchors={len(self.keypoint_map)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[CameraTracker] Could not initialize Serverless court model: {exc}. "
                    "Falling back to optical flow.",
                    flush=True,
                )
            return

        if self.keypoint_backend not in {"legacy", "local"}:
            print(
                f"[CameraTracker] Unsupported BASKETBALL_COURT_INFERENCE_BACKEND="
                f"{self.keypoint_backend!r}; using optical-flow fallback.",
                flush=True,
            )
            return
        if not LOCAL_INFERENCE_AVAILABLE or get_local_inference_model is None:
            print(
                "[CameraTracker] legacy inference package unavailable; using optical-flow fallback. "
                "For the lightweight hosted path set BASKETBALL_COURT_INFERENCE_BACKEND=serverless.",
                flush=True,
            )
            return
        api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
        if not api_key:
            print("[CameraTracker] ROBOFLOW_API_KEY not set; using optical-flow fallback.")
            return
        try:
            self.keypoint_model = get_local_inference_model(model_id=self.model_id)
            self.use_keypoint_model = True
            print(
                f"[CameraTracker] Loaded local court keypoint model: {self.model_id} "
                f"schema={self.keypoint_schema} anchors={len(self.keypoint_map)}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[CameraTracker] Could not load keypoint model: {exc}. Falling back to optical flow.",
                flush=True,
            )

    def _init_first_frame(self, frame: np.ndarray):
        self.frame_h, self.frame_w = frame.shape[:2]
        detected_H, detected_poly = self._detect_keypoints(frame)

        if self.base_polygon is None:
            self.base_polygon = (
                detected_poly
                if detected_poly is not None
                else np.array(
                    [[0, 0], [self.frame_w, 0], [self.frame_w, self.frame_h], [0, self.frame_h]],
                    dtype=np.int32,
                )
            )

        if self.initial_H is None:
            self.initial_H = detected_H if detected_H is not None else np.eye(3, dtype=np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
        )

    @staticmethod
    def _extract_keypoints(results: Any) -> list[Any]:
        if hasattr(results, "predictions") and results.predictions:
            return list(getattr(results.predictions[0], "keypoints", []) or [])
        if hasattr(results, "keypoints"):
            return list(results.keypoints or [])
        if isinstance(results, dict) and results.get("predictions"):
            first_prediction = results["predictions"][0]
            if isinstance(first_prediction, dict):
                return list(first_prediction.get("keypoints", []) or [])
            return list(getattr(first_prediction, "keypoints", []) or [])
        return []

    def _detect_keypoints(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.use_keypoint_model or self.keypoint_model is None:
            self.last_calibration_diagnostics = {
                "valid": False,
                "anchor_count": 0,
                "inlier_count": 0,
                "reprojection_rmse_px": None,
                "source": "keypoint_model_unavailable",
                "keypoint_backend": self.keypoint_backend,
            }
            return None, None

        try:
            raw_results = self.keypoint_model.infer(frame, confidence=self.confidence)
            results = raw_results[0] if isinstance(raw_results, list) else raw_results
            keypoints_data = self._extract_keypoints(results)
            src_points: list[list[float]] = []
            dst_points: list[tuple[float, float]] = []

            for fallback_idx, kp in enumerate(keypoints_data):
                if isinstance(kp, dict):
                    x, y = kp.get("x"), kp.get("y")
                    conf = kp.get("confidence", 1.0)
                    idx = kp.get("id", kp.get("class_id", fallback_idx))
                else:
                    x, y = getattr(kp, "x", None), getattr(kp, "y", None)
                    conf = getattr(kp, "confidence", 1.0)
                    idx = getattr(kp, "id", getattr(kp, "class_id", fallback_idx))
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    idx = fallback_idx

                if (
                    x is not None
                    and y is not None
                    and float(conf) >= self.anchor_confidence
                    and idx in self.keypoint_map
                ):
                    src_points.append([float(x), float(y)])
                    dst_points.append(self.keypoint_map[idx])

            estimate = estimate_homography(
                src_points,
                dst_points,
                max_rmse_px=self.max_reprojection_rmse_px,
            )
            self.last_calibration_diagnostics = {
                **estimate.diagnostics(),
                "source": "direct_keypoints",
                "model_id": self.model_id,
                "keypoint_schema": self.keypoint_schema,
                "keypoint_map_size": len(self.keypoint_map),
                "keypoint_backend": self.keypoint_backend,
            }

            court_poly = None
            if len(src_points) >= 3:
                hull = cv2.convexHull(np.asarray(src_points, dtype=np.int32))
                court_poly = hull.reshape(-1, 2)
            return (estimate.matrix if estimate.valid else None), court_poly
        except Exception as exc:
            self.last_calibration_diagnostics = {
                "valid": False,
                "anchor_count": 0,
                "inlier_count": 0,
                "reprojection_rmse_px": None,
                "source": "keypoint_exception",
                "keypoint_backend": self.keypoint_backend,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[CameraTracker] Keypoint detection failed: {exc}")
            return None, None

    def reset_flow(self):
        self.prev_gray = None
        self.prev_pts = None
        self.M_accum = np.eye(3, dtype=np.float32)

    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.prev_gray is None:
            self._init_first_frame(frame)
            return self._get_current_metrics()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        direct_H, direct_poly = self._detect_keypoints(frame)
        if direct_H is not None:
            self.initial_H = direct_H
            self.M_accum = np.eye(3, dtype=np.float32)
            if direct_poly is not None:
                self.base_polygon = direct_poly
            self.prev_gray = gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7
            )
            return self._get_current_metrics()

        if self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, winSize=(21, 21), maxLevel=3
            )
            if next_pts is not None and status is not None:
                good_old = self.prev_pts[status == 1]
                good_new = next_pts[status == 1]
                if len(good_old) > 30:
                    M, mask = cv2.findHomography(
                        good_new, good_old, cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if M is not None:
                        inliers = int(np.sum(mask)) if mask is not None else 0
                        translation = float(abs(M[0, 2]) + abs(M[1, 2]))
                        if inliers > 20 and translation < 150:
                            self.M_accum = np.dot(self.M_accum, M)
                            self.last_calibration_diagnostics["source"] = "optical_flow_hold"
                self.prev_pts = (
                    cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)
                    if len(good_new) < 50
                    else good_new.reshape(-1, 1, 2)
                )
        self.prev_gray = gray.copy()
        return self._get_current_metrics()

    def _get_current_metrics(self) -> Tuple[np.ndarray, np.ndarray]:
        initial = self.initial_H if self.initial_H is not None else np.eye(3, dtype=np.float32)
        current_H = np.dot(initial, self.M_accum)
        try:
            warped = cv2.perspectiveTransform(
                np.array([self.base_polygon], dtype=np.float32), np.linalg.inv(self.M_accum)
            )
            current_polygon = np.int32(warped)[0]
        except Exception:
            current_polygon = self.base_polygon
        return current_H, current_polygon

    def transform_feet_to_2d_court(
        self, foot_xy: Tuple[float, float], H: Optional[np.ndarray] = None
    ) -> Optional[Tuple[float, float]]:
        homography = H if H is not None else self._get_current_metrics()[0]
        return project_image_point_to_court(foot_xy, homography)

    def is_on_court_meters(
        self,
        foot_xy: Tuple[float, float],
        H: Optional[np.ndarray] = None,
        margin_meters: float = 0.5,
    ) -> bool:
        coords = self.transform_feet_to_2d_court(foot_xy, H)
        if coords is None:
            return True
        x_m, y_m = coords
        return (-margin_meters <= x_m <= 28.0 + margin_meters) and (
            -margin_meters <= y_m <= 15.0 + margin_meters
        )

    def get_backboard_roi_from_homography(
        self,
        current_H: np.ndarray,
        roi_width_ratio: float = 0.12,
        roi_height_ratio: float = 0.22,
    ) -> Optional[Tuple[int, int, int, int]]:
        if current_H is None or np.allclose(current_H, np.eye(3)):
            return None
        try:
            H_inv = np.linalg.inv(current_H)
            for hoop_m in ((1.575, 7.5), (26.425, 7.5)):
                projected = project_image_point_to_court(hoop_m, H_inv)
                if projected is None:
                    continue
                u, v = projected
                if 0 <= u <= self.frame_w and 0 <= v <= self.frame_h:
                    roi_w = int(self.frame_w * roi_width_ratio)
                    roi_h = int(self.frame_h * roi_height_ratio)
                    x0 = max(0, int(u - roi_w / 2.0))
                    y0 = max(0, int(v - roi_h * 0.75))
                    return (x0, y0, min(self.frame_w - x0, roi_w), min(self.frame_h - y0, roi_h))
        except Exception:
            return None
        return None

    def project_entities(self, frame_state: Any) -> Any:
        annotate_projected_entities(frame_state)
        return frame_state

    def process(self, frame_state) -> Any:
        current_H, current_polygon = self.update(frame_state.frame)
        frame_state.homography_matrix = current_H
        frame_state.camera_polygon = current_polygon
        frame_state.backboard_roi = self.get_backboard_roi_from_homography(current_H)
        camera_diag = frame_state.diagnostics.setdefault("camera", {})
        camera_diag.update(self.last_calibration_diagnostics)
        camera_diag.update(
            {
                "keypoint_model_active": bool(self.use_keypoint_model),
                "keypoint_backend": self.keypoint_backend,
                "homography_is_identity": bool(np.allclose(current_H, np.eye(3))),
                "diagnostic_only": True,
            }
        )
        return frame_state
