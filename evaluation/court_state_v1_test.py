from __future__ import annotations

import cv2
import numpy as np

from core.frame_state import FrameState
from services.pipeline import Pipeline
from services.vision.court_state_v1 import CourtStateEstimatorV1


def _court_frame() -> np.ndarray:
    hsv = np.zeros((240, 320, 3), dtype=np.uint8)
    hsv[:, :, 0] = 15
    hsv[:, :, 1] = 180
    hsv[:, :, 2] = 200
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Synthetic sideline/non-court strip for an explicit off-court entity.
    frame[:, :72] = 0
    return frame


def _player(track_id: int, box: list[int], team: str = "FC Porto") -> dict:
    return {
        "track_id": track_id,
        "box": box,
        "center": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
        "confidence": 0.9,
        "team": team,
        "team_confidence": 0.8,
        "team_margin": 0.2,
        "shirt_num": "8",
        "player_name": "Synthetic Player",
    }


def test_image_space_court_state_and_identity_gate() -> None:
    frame = _court_frame()
    state = FrameState(frame_id=12, timestamp=0.48, frame=frame)
    state.players = [
        _player(1, [10, 70, 55, 190], team="Ovarense"),  # explicit non-court strip
        _player(2, [90, 80, 125, 180]),
        _player(3, [130, 75, 165, 180]),
        _player(4, [175, 78, 210, 182]),
        _player(5, [220, 82, 255, 184]),
    ]
    state.referees = [
        {"track_id": 21, "box": [260, 80, 290, 184], "confidence": 0.93}
    ]

    estimator = CourtStateEstimatorV1()
    court = estimator.update(state, gameplay_diagnostics={"court_ratio": 0.75})
    state.court_state = court

    assert court.mode == "image_space_fallback"
    assert court.geometry_valid is False
    assert court.court_visible is True
    assert court.view_mode == "wide"
    assert court.player_count == 5
    assert court.on_court_player_count == 4
    assert court.analysis_eligible is True
    assert state.players[0]["active_on_court"] is False
    assert state.players[0]["identity_eligible"] is False
    assert state.players[1]["active_on_court"] is True

    Pipeline._apply_court_identity_gate(state)
    assert state.players[0]["team"] is None
    assert state.players[0]["shirt_num"] is None
    assert state.players[0]["player_name"] is None
    assert state.players[1]["team"] == "FC Porto"


def test_closeup_is_not_metric_eligible() -> None:
    frame = _court_frame()
    state = FrameState(frame_id=13, timestamp=0.52, frame=frame)
    state.players = [
        _player(1, [85, 20, 155, 225]),
        _player(2, [170, 15, 245, 225]),
    ]

    estimator = CourtStateEstimatorV1()
    court = estimator.update(state, gameplay_diagnostics={"court_ratio": 0.75})

    assert court.court_visible is True
    assert court.view_mode == "tight"
    assert court.analysis_eligible is False
    assert court.eligibility_reason == "not_wide_basketball_view"


def test_calibrated_mode_uses_metric_bounds() -> None:
    frame = _court_frame()
    state = FrameState(frame_id=14, timestamp=0.56, frame=frame)
    # Simple synthetic pixel -> FIBA metre mapping.
    state.homography_matrix = np.array(
        [
            [28.0 / 320.0, 0.0, 0.0],
            [0.0, 15.0 / 240.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    state.diagnostics["camera"] = {
        "valid": True,
        "homography_is_identity": False,
        "source": "synthetic_test",
        "reprojection_rmse_px": 1.5,
    }
    state.players = [
        _player(1, [90, 80, 125, 180]),
        _player(2, [130, 75, 165, 180]),
        _player(3, [175, 78, 210, 182]),
        _player(4, [220, 82, 255, 184]),
    ]

    estimator = CourtStateEstimatorV1()
    court = estimator.update(state, gameplay_diagnostics={"court_ratio": 0.75})

    assert court.mode == "calibrated"
    assert court.geometry_valid is True
    assert court.on_court_player_count == 4
    assert court.analysis_eligible is True
    assert all(player.get("court_xy") is not None for player in state.players)


def main() -> None:
    test_image_space_court_state_and_identity_gate()
    test_closeup_is_not_metric_eligible()
    test_calibrated_mode_uses_metric_bounds()
    print("court_state_v1_test: PASS")


if __name__ == "__main__":
    main()
