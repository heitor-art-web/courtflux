from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.frame_state import FrameState
from services.diagnostics import RunDiagnostics
from services.vision.court_state_v1 import CourtStateEstimatorV1
from services.vision.hypothesis_p0_detector import PerformanceDetector


def _court_frame() -> np.ndarray:
    hsv = np.zeros((240, 320, 3), dtype=np.uint8)
    hsv[:, :, 0] = 15
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 200
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame[:, :72] = 0
    return frame


def _player(track_id: int, box: list[int]) -> dict:
    return {
        "track_id": track_id,
        "box": box,
        "confidence": 0.9,
        "team": None,
    }


def test_identity_gate_runs_before_team_jersey_evidence() -> None:
    state = FrameState(frame_id=1, timestamp=0.04, frame=_court_frame())
    state.players = [
        _player(10, [10, 90, 55, 190]),   # outside calibrated FIBA bounds
        _player(11, [90, 80, 125, 180]),
        _player(12, [130, 78, 165, 180]),
        _player(13, [175, 82, 210, 182]),
        _player(14, [220, 84, 255, 184]),
    ]

    # Calibrated synthetic image -> court mapping. The visible court begins around
    # x=72 px, so the first player projects outside the FIBA court while the remaining
    # four project inside. OFFCOURT is allowed only in this calibrated mode; image-space
    # fallback now abstains instead of forcing False.
    x_scale = 28.0 / (320.0 - 72.0)
    state.homography_matrix = np.array(
        [
            [x_scale, 0.0, -72.0 * x_scale],
            [0.0, 15.0 / 240.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    state.diagnostics["camera"] = {
        "valid": True,
        "homography_is_identity": False,
        "source": "synthetic_test",
        "reprojection_rmse_px": 1.0,
    }

    detector = PerformanceDetector.__new__(PerformanceDetector)
    detector.identity_court_estimator = CourtStateEstimatorV1()
    detector._prepare_identity_court_gate(state)

    assert state.court_state is not None
    assert state.court_state.mode == "calibrated"
    assert state.court_state.analysis_eligible is True
    assert state.players[0]["active_on_court"] is False
    assert all(player["active_on_court"] is True for player in state.players[1:])
    assert [p["track_id"] for p in detector._identity_eligible_players(state)] == [
        11,
        12,
        13,
        14,
    ]

    diag = state.diagnostics["detector"]
    assert diag["identity_court_gate_prepared"] is True
    assert diag["identity_frame_analysis_eligible"] is True
    assert diag["identity_players_total"] == 5
    assert diag["identity_players_explicit_off_court"] == 1
    assert diag["identity_players_eligible"] == 4

    tight = FrameState(frame_id=2, timestamp=0.08, frame=_court_frame())
    tight.players = [
        _player(20, [85, 20, 155, 225]),
        _player(21, [170, 15, 245, 225]),
    ]
    tight_detector = PerformanceDetector.__new__(PerformanceDetector)
    tight_detector.identity_court_estimator = CourtStateEstimatorV1()
    tight_detector._prepare_identity_court_gate(tight)
    assert tight.court_state is not None
    assert tight.court_state.analysis_eligible is False
    assert tight_detector._identity_eligible_players(tight) == []
    assert tight.diagnostics["detector"]["identity_players_eligible"] == 0


def _diagnostic_state(
    frame_id: int,
    *,
    eligible: bool,
    track_id: int,
    team_confidence: float,
) -> FrameState:
    state = FrameState(
        frame_id=frame_id,
        timestamp=frame_id / 25.0,
        frame=np.zeros((100, 160, 3), dtype=np.uint8),
    )
    state.valid_gameplay = eligible
    state.players = [
        {
            "track_id": track_id,
            "box": [20, 20, 50, 80],
            "confidence": 0.9,
            "team": "FC Porto",
            "team_confidence": team_confidence,
            "team_margin": 0.1,
            "shirt_num": str(track_id),
        }
    ]
    state.diagnostics["detector"] = {
        "raw_total": 2,
        "raw_player_candidates": 1,
        "accepted_player_candidates": 1,
        "parsed_players": 1,
        "raw_ball_candidates": 1,
        "accepted_ball_candidates": 1,
        "selected_ball": True,
        "selected_ball_conf": 0.8,
        "selected_ball_center": [20.0 + frame_id, 30.0],
        "ball_association": {
            "hold_age_frames": 0,
            "rejected_candidates": 0,
        },
        "jersey_attempts": 1,
        "jersey_successes": 1,
        "jersey_predictions": [
            {
                "raw_number": str(track_id),
                "candidate_number": str(track_id),
                "rejection_reason": "accepted",
            }
        ],
        "team_predictions": [],
        "timing_ms": {"parse": 1.0},
        "inference_conf": 0.03,
        "player_conf": 0.40,
        "ball_conf": 0.075,
    }
    state.diagnostics["analytics"] = {
        "ball_source": "detector",
        "players_final": 1,
    }
    state.diagnostics["court_state"] = {
        "analysis_eligible": eligible,
        "eligibility_reason": (
            "court_and_entities_consistent" if eligible else "not_wide_basketball_view"
        ),
    }
    return state


def test_diagnostics_scope_and_ball_segments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        diagnostics = RunDiagnostics(
            root=Path(tmp),
            video_source="synthetic.mp4",
            start_time=0.0,
            end_time=1.0,
            model_path="weights/model_v1.1.0.pt",
            detector_conf=0.40,
            imgsz=960,
            device="cpu",
            tracker="botsort.yaml",
            detection_only=True,
            enable_jersey=True,
            batch_size=20,
        )

        diagnostics.log_frame(
            _diagnostic_state(1, eligible=True, track_id=8, team_confidence=0.8)
        )
        diagnostics.log_frame(
            _diagnostic_state(2, eligible=True, track_id=8, team_confidence=0.8)
        )
        diagnostics.log_frame(
            _diagnostic_state(3, eligible=False, track_id=99, team_confidence=0.01)
        )
        diagnostics.log_frame(
            _diagnostic_state(4, eligible=True, track_id=9, team_confidence=0.6)
        )

        summary_path = diagnostics.finish()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in diagnostics.frames_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    assert summary["total_frames"] == 4
    assert summary["metric_eligible_frames"] == 3
    assert summary["metric_scope"]["frames_with_detector_inference"] == 4
    assert summary["metric_scope"]["metric_excluded_frames"] == 1
    assert summary["metric_scope"]["metric_eligible_rate"] == 75.0

    identity = summary["players"]["identity_diagnostics"]
    assert identity["scope"] == "metric_eligible_frames_only"
    assert identity["identified_tracks"] == 2
    assert identity["avg_team_confidence"] == 0.7333
    assert summary["players"]["jersey_attempts"] == 3
    assert identity["jersey_attempts_all_inferred_frames"] == 4
    assert "99" not in identity["raw_prediction_counts"]

    continuity = summary["ball"]["continuity"]
    assert continuity["segment_boundary_policy"] == "reset_on_metric_ineligible_frame"
    assert continuity["detection_streak_count"] == 2
    assert continuity["detection_streak_max_frames"] == 2

    trajectory = summary["ball"]["trajectory"]
    assert trajectory["trajectory_streak_count"] == 2
    assert trajectory["trajectory_streak_max_frames"] == 2

    assert records[2]["metric_eligible"] is False
    assert records[2]["metric_scope_reason"] == "not_wide_basketball_view"


def main() -> None:
    test_identity_gate_runs_before_team_jersey_evidence()
    test_diagnostics_scope_and_ball_segments()
    print("p1_correctness_cleanup_test: PASS")


if __name__ == "__main__":
    main()
