from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from core.ball_state import BallStateStatus
from core.frame_state import FrameState
from services.vision.ball_state_v1 import BallStateEstimatorV1
from services.vision.court_state_v1 import CourtStateEstimatorV1
from services.vision.player_state_v1 import PlayerStateEstimatorV1
from services.vision.single_forward_p0_detector import PerformanceDetector as SingleForwardDetector
from services.vision.single_forward_tracker import SingleForwardEntityTracker


def _court_frame() -> np.ndarray:
    hsv = np.zeros((240, 320, 3), dtype=np.uint8)
    hsv[:, :, 0] = 15
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 200
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_player_state_accepts_runtime_court_xy() -> None:
    estimator = PlayerStateEstimatorV1()
    states = estimator.update(
        [
            {
                "class_name": "player",
                "track_id": 7,
                "box": [10, 20, 50, 100],
                "confidence": 0.8,
                "court_xy": [5.5, 7.0],
                "court_xy_valid": True,
                "active_on_court": True,
                "court_eligibility_source": "calibrated_fiba_bounds",
            },
            {
                "class_name": "player",
                "track_id": 70,
                "box": [5, 20, 40, 100],
                "confidence": 0.9,
                "active_on_court": False,
                "court_eligibility_source": "image_space_off_court_surface",
            },
            {
                "class_name": "referee",
                "track_id": 8,
                "box": [60, 20, 100, 100],
                "confidence": 0.9,
            },
        ],
        frame_id=10,
        timestamp_sec=0.4,
    )
    assert len(states) == 1
    assert states[0].track_id == 7
    assert states[0].court_position_m == (5.5, 7.0)
    assert states[0].provenance["active_on_court"] is True
    assert estimator.last_diagnostics["explicit_off_court_excluded"] == 1
    assert estimator.last_diagnostics["referee_excluded"] == 1


def test_ball_state_runtime_observed_predicted_missing() -> None:
    estimator = BallStateEstimatorV1()
    observed = estimator.from_runtime_evidence(
        {
            "box": [10, 10, 20, 20],
            "center": [15.0, 15.0],
            "confidence": 0.8,
            "interpolated": False,
        },
        {"source": "detector_temporal_association", "selected_index": 0},
        frame_id=1,
        timestamp_sec=0.04,
    )
    assert observed.status is BallStateStatus.OBSERVED

    observed_2 = estimator.from_runtime_evidence(
        {
            "box": [12, 10, 22, 20],
            "center": [17.0, 15.0],
            "confidence": 0.75,
            "interpolated": False,
        },
        {"source": "detector_temporal_association", "selected_index": 0},
        frame_id=2,
        timestamp_sec=0.08,
    )
    assert observed_2.velocity_px_per_frame == (2.0, 0.0)

    predicted = estimator.from_runtime_evidence(
        {
            "box": [14, 10, 24, 20],
            "center": [19.0, 15.0],
            "confidence": 0.6,
            "interpolated": True,
        },
        {"source": "predicted_hold", "hold_age_frames": 1},
        frame_id=3,
        timestamp_sec=0.12,
    )
    assert predicted.status is BallStateStatus.PREDICTED
    assert predicted.velocity_px_per_frame == (2.0, 0.0)

    missing = estimator.from_runtime_evidence(
        None,
        {"source": "missing", "hold_age_frames": 6},
        frame_id=4,
        timestamp_sec=0.16,
    )
    assert missing.status is BallStateStatus.MISSING
    assert missing.position() is None


def test_court_state_keeps_rim_backboard_as_geometry_evidence() -> None:
    frame = _court_frame()
    state = FrameState(frame_id=1, timestamp=0.04, frame=frame)
    state.players = [
        {
            "track_id": index,
            "box": [40 + index * 35, 70, 65 + index * 35, 180],
            "confidence": 0.9,
        }
        for index in range(4)
    ]
    state.diagnostics["detector"] = {
        "p1_detections": [
            {
                "class_name": "rim",
                "confidence": 0.95,
                "box": [250, 60, 285, 75],
            },
            {
                "class_name": "rim",
                "confidence": 0.20,
                "box": [210, 60, 240, 75],
            },
            {
                "class_name": "backboard",
                "confidence": 0.6,
                "box": [245, 30, 300, 80],
            },
            {
                "class_name": "backboard",
                "confidence": 0.10,
                "box": [180, 30, 220, 80],
            },
        ]
    }
    court = CourtStateEstimatorV1().update(
        state,
        gameplay_diagnostics={"court_ratio": 0.5},
    )
    assert {item["kind"] for item in court.basket_evidence} == {"rim", "backboard"}
    assert len(court.basket_evidence) == 2
    assert all(item["event_truth"] is False for item in court.basket_evidence)
    assert {item["threshold"] for item in court.basket_evidence} == {0.4, 0.15}


def test_single_forward_tracker_constructor_matches_installed_ultralytics() -> None:
    tracker = SingleForwardEntityTracker(
        "botsort.yaml",
        device="cpu",
        frame_rate=25.0,
    )
    metadata = tracker.metadata()
    assert metadata["detector_inference_owned_by_tracker"] is False
    assert metadata["raw_ball_evidence_passes_through_tracker"] is False
    assert tracker._rows_from_array(None) == []
    assert tracker._rows_from_array(np.empty((0, 8), dtype=np.float32)) == []


class _FakeSingleForwardDetector(SingleForwardDetector):
    def __init__(self) -> None:
        self.model = SimpleNamespace(
            names={0: "basketball", 1: "player", 2: "referee", 3: "rim", 4: "backboard"}
        )
        self.player_conf_thresh = 0.40
        self.referee_conf_thresh = 0.50
        self.ball_conf_thresh = 0.075
        self.ball_temporal_enabled = False
        self.ball_candidate_max_det = 100
        self.ball_candidate_class_ids = [0]
        self.split_ball_detection = True
        self.inference_conf_thresh = 0.03
        self.detector_conf = 0.40
        self.imgsz = 960
        self.iou = 0.5
        self.tracker = "botsort.yaml"
        self.device = "cpu"
        self.enable_jersey = False
        self.jersey_interval_frames = 24
        self.ball_tracker = SimpleNamespace(
            score_confidence_weight=0.58,
            score_proximity_weight=0.32,
            score_size_weight=0.10,
            base_gate_norm=0.12,
            gate_growth_per_frame=0.025,
            max_gate_norm=0.28,
            high_confidence_gate_threshold=0.85,
            high_confidence_gate_multiplier=1.35,
            velocity_alpha=0.4,
            max_hold_frames=5,
            reset_after_frames=12,
            reacquire_enabled=False,
            reacquire_min_confidence=0.45,
            reacquire_confidence_margin=0.15,
            reacquire_confirm_frames=2,
            reacquire_radius_norm=0.04,
            reacquire_min_size_similarity=0.25,
        )

    @staticmethod
    def _is_referee_class(name: str) -> bool:
        return "referee" in str(name).lower()

    def _select_ball(self, candidates, frame_state):
        if not candidates:
            return None, {
                "source": "missing",
                "selected_index": None,
                "rejected_candidates": 0,
                "hold_age_frames": 0,
            }
        ball = dict(candidates[0])
        ball["source"] = "detector_highest_conf"
        return ball, {
            "source": "detector_highest_conf",
            "selected_index": 0,
            "rejected_candidates": 0,
            "hold_age_frames": 0,
        }

    def ball_candidate_backend_metadata(self):
        return {"backend": "yolo_shared", "shared_primary_model": True}

    def _tracker_config(self):
        return {
            "reacquire": {},
            "candidate_handoff": {"source": "yolo_single_forward_raw"},
        }


def test_single_forward_parser_routes_classes_once() -> None:
    detector = _FakeSingleForwardDetector()
    state = FrameState(
        frame_id=1,
        timestamp=0.04,
        frame=np.zeros((240, 320, 3), dtype=np.uint8),
    )
    tracked_rows = [
        {
            "box": [20, 40, 60, 180],
            "track_id": 10,
            "confidence": 0.9,
            "class_id": 1,
        },
        {
            "box": [80, 40, 120, 180],
            "track_id": 20,
            "confidence": 0.8,
            "class_id": 2,
        },
    ]
    raw_rows = [
        {"detection_index": 0, "class_id": 1, "class_name": "player", "confidence": 0.9, "box": [20, 40, 60, 180]},
        {"detection_index": 1, "class_id": 2, "class_name": "referee", "confidence": 0.8, "box": [80, 40, 120, 180]},
        {"detection_index": 2, "class_id": 0, "class_name": "basketball", "confidence": 0.7, "box": [150, 90, 160, 100]},
        {"detection_index": 3, "class_id": 3, "class_name": "rim", "confidence": 0.95, "box": [250, 50, 285, 65]},
    ]
    ball_candidates = [
        {
            "box": [150, 90, 160, 100],
            "center": [155, 95],
            "confidence": 0.7,
            "detector_confidence": 0.7,
            "source": "full_frame",
        }
    ]
    detector._parse_single_forward(state, tracked_rows, ball_candidates, raw_rows)
    assert len(state.players) == 1
    assert state.players[0]["track_id"] == 10
    assert len(state.referees) == 1
    assert state.referees[0]["track_id"] == 20
    assert state.ball is not None
    diag = state.diagnostics["detector"]
    assert diag["single_forward"] is True
    assert diag["raw_class_counts"]["rim"] == 1
    assert len(diag["p1_detections"]) == 4


def main() -> None:
    test_player_state_accepts_runtime_court_xy()
    test_ball_state_runtime_observed_predicted_missing()
    test_court_state_keeps_rim_backboard_as_geometry_evidence()
    test_single_forward_tracker_constructor_matches_installed_ultralytics()
    test_single_forward_parser_routes_classes_once()
    print("p1_runtime_consolidation_test: PASS")


if __name__ == "__main__":
    main()
