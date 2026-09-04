import cv2
import numpy as np
import os
from typing import Tuple, Optional, Any
from dotenv import load_dotenv

load_dotenv()
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

try:
    from inference import get_model
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


# Official 14-Keypoint FIBA Court Reference Map (28m x 15m in meters)
FIBA_COURT_KEYPOINTS_14 = {
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


class CameraTracker:
    def __init__(
        self,
        first_frame: Optional[np.ndarray] = None,
        initial_H: Optional[np.ndarray] = None,
        initial_polygon: Optional[np.ndarray] = None,
        use_keypoint_model: bool = True,
        model_id: str = "basketball-court-detection-2/14",
        confidence: float = 0.3,
        anchor_confidence: float = 0.5,
    ):
        self.frame_h = 0
        self.frame_w = 0

        self.confidence = confidence
        self.anchor_confidence = anchor_confidence
        self.keypoint_map = FIBA_COURT_KEYPOINTS_14
        self.keypoint_model = None

        requested_keypoints = bool(use_keypoint_model)
        self.use_keypoint_model = (
            requested_keypoints and INFERENCE_AVAILABLE and bool(roboflow_api_key)
        )

        if requested_keypoints and not INFERENCE_AVAILABLE:
            print("[CameraTracker] inference package unavailable; using optical-flow fallback.")
        elif requested_keypoints and not roboflow_api_key:
            print("[CameraTracker] ROBOFLOW_API_KEY not set; using optical-flow fallback.")

        if self.use_keypoint_model:
            try:
                self.keypoint_model = get_model(model_id=model_id)
                print(f"[CameraTracker] Loaded court keypoint model: {model_id}")
            except Exception as e:
                print(f"[CameraTracker] Could not load keypoint model: {e}. Falling back to optical flow.")
                self.use_keypoint_model = False

        self.initial_H = initial_H
        self.base_polygon = initial_polygon
        self.M_accum = np.eye(3, dtype=np.float32)
        self.prev_gray = None
        self.prev_pts = None

        if first_frame is not None:
            self._init_first_frame(first_frame)

    def _init_first_frame(self, frame: np.ndarray):
        """Initialize dimensions, initial homography, polygon, and optical-flow anchors."""
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
            gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

    def _detect_keypoints(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Infer court keypoints and estimate image→court homography plus visible polygon."""
        if not self.use_keypoint_model or self.keypoint_model is None:
            return None, None

        try:
            raw_results = self.keypoint_model.infer(frame, confidence=self.confidence)
            results = raw_results[0] if isinstance(raw_results, list) else raw_results

            keypoints_data = []
            if hasattr(results, "predictions") and results.predictions:
                keypoints_data = getattr(results.predictions[0], "keypoints", [])
            elif hasattr(results, "keypoints"):
                keypoints_data = results.keypoints or []
            elif isinstance(results, dict) and "predictions" in results:
                keypoints_data = results["predictions"][0].get("keypoints", [])

            if not keypoints_data:
                return None, None

            src_points = []
            dst_points = []

            for idx, kp in enumerate(keypoints_data):
                if isinstance(kp, dict):
                    x, y = kp.get("x"), kp.get("y")
                    conf = kp.get("confidence", 1.0)
                else:
                    x, y = getattr(kp, "x", None), getattr(kp, "y", None)
                    conf = getattr(kp, "confidence", 1.0)

                if (
                    x is not None
                    and y is not None
                    and conf >= self.anchor_confidence
                    and idx in self.keypoint_map
                ):
                    src_points.append([x, y])
                    dst_points.append(self.keypoint_map[idx])

            H_matrix = None
            if len(src_points) >= 4:
                src_pts = np.array(src_points, dtype=np.float32)
                dst_pts = np.array(dst_points, dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    H_matrix = H

            court_poly = None
            if len(src_points) >= 3:
                hull = cv2.convexHull(np.array(src_points, dtype=np.int32))
                court_poly = hull.reshape(-1, 2)

            return H_matrix, court_poly

        except Exception as e:
            print(f"[CameraTracker] Keypoint detection failed: {e}")
            return None, None

    def reset_flow(self):
        """Reset tracking references after a commercial or hard broadcast cut."""
        self.prev_gray = None
        self.prev_pts = None
        self.M_accum = np.eye(3, dtype=np.float32)

    def update(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update camera geometry using keypoint re-anchoring plus optical-flow fallback."""
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
                gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=7,
            )
            return self._get_current_metrics()

        if self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )

            if next_pts is not None and status is not None:
                good_old = self.prev_pts[status == 1]
                good_new = next_pts[status == 1]

                if len(good_old) > 30:
                    M, mask = cv2.findHomography(
                        good_new,
                        good_old,
                        cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                    )

                    if M is not None:
                        inliers = np.sum(mask) if mask is not None else 0
                        translation_magnitude = np.abs(M[0, 2]) + np.abs(M[1, 2])

                        if inliers > 20 and translation_magnitude < 150:
                            self.M_accum = np.dot(self.M_accum, M)

                if len(good_new) < 50:
                    self.prev_pts = cv2.goodFeaturesToTrack(
                        gray,
                        maxCorners=200,
                        qualityLevel=0.01,
                        minDistance=10,
                        blockSize=7,
                    )
                else:
                    self.prev_pts = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()
        return self._get_current_metrics()

    def _get_current_metrics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate current homography and warped visible-court polygon."""
        current_H_metric = np.dot(self.initial_H, self.M_accum)

        try:
            M_accum_inv = np.linalg.inv(self.M_accum)
            pts = np.array([self.base_polygon], dtype=np.float32)
            warped_pts = cv2.perspectiveTransform(pts, M_accum_inv)
            current_polygon_px = np.int32(warped_pts)[0]
        except Exception:
            current_polygon_px = self.base_polygon

        return current_H_metric, current_polygon_px

    def transform_feet_to_2d_court(
        self,
        foot_xy: Tuple[float, float],
        H: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float]]:
        """Map a player's bounding-box bottom-center to FIBA court meters."""
        homography = H if H is not None else np.dot(self.initial_H, self.M_accum)
        if homography is None:
            return None

        try:
            point = np.array([foot_xy[0], foot_xy[1], 1.0], dtype=np.float32).reshape(3, 1)
            world_pt = np.dot(homography, point)
            if world_pt[2] == 0:
                return None

            x_meter = world_pt[0][0] / world_pt[2][0]
            y_meter = world_pt[1][0] / world_pt[2][0]
            return float(x_meter), float(y_meter)
        except Exception:
            return None

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
        """Project FIBA hoop locations into image space and return a visible-hoop ROI."""
        if current_H is None or np.array_equal(current_H, np.eye(3)):
            return None

        try:
            H_inv = np.linalg.inv(current_H)
            hoops_world = [
                np.array([1.575, 7.5, 1.0], dtype=np.float32),
                np.array([26.425, 7.5, 1.0], dtype=np.float32),
            ]

            visible_hoop_px = None
            for hoop_m in hoops_world:
                px_homo = np.dot(H_inv, hoop_m.reshape(3, 1))
                if px_homo[2][0] == 0:
                    continue

                u = px_homo[0][0] / px_homo[2][0]
                v = px_homo[1][0] / px_homo[2][0]
                if 0 <= u <= self.frame_w and 0 <= v <= self.frame_h:
                    visible_hoop_px = (u, v)
                    break

            if visible_hoop_px is None:
                return None

            u_c, v_c = visible_hoop_px
            roi_w = int(self.frame_w * roi_width_ratio)
            roi_h = int(self.frame_h * roi_height_ratio)
            x0 = max(0, int(u_c - (roi_w / 2.0)))
            y0 = max(0, int(v_c - (roi_h * 0.75)))
            w0 = min(self.frame_w - x0, roi_w)
            h0 = min(self.frame_h - y0, roi_h)
            return (x0, y0, w0, h0)

        except Exception:
            return None

    def process(self, frame_state) -> Any:
        """Update dynamic homography, court polygon, and hoop ROI in FrameState."""
        current_H, current_polygon = self.update(frame_state.frame)
        frame_state.homography_matrix = current_H
        frame_state.camera_polygon = current_polygon
        frame_state.backboard_roi = self.get_backboard_roi_from_homography(current_H)

        frame_state.diagnostics.setdefault("camera", {})
        frame_state.diagnostics["camera"].update(
            {
                "keypoint_model_active": bool(self.use_keypoint_model),
                "homography_is_identity": bool(np.allclose(current_H, np.eye(3))),
            }
        )
        return frame_state
