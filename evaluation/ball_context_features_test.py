from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_context_features import BallContextFeatureEnricher


def _state(frame_id: int, center: tuple[float, float]) -> SimpleNamespace:
    hypothesis = {
        "tracker_id": 7,
        "hypothesis_id": "ocsort:7",
        "lifecycle": "TRACKED",
        "observed": True,
        "center": list(center),
        "box": [center[0] - 1, center[1] - 1, center[0] + 1, center[1] + 1],
    }
    return SimpleNamespace(
        frame_id=frame_id,
        valid_gameplay=True,
        homography_matrix=np.eye(3, dtype=np.float64),
        players=[
            {
                "track_id": 11,
                "box": [8.0, 4.0, 12.0, 8.0],
                "court_xy": [10.0, 8.0],
            }
        ],
        diagnostics={
            "camera": {
                "valid": True,
                "source": "direct_keypoints",
                "anchor_count": 12,
                "inlier_count": 6,
                "reprojection_rmse_px": 1.5261,
            },
            "detector": {
                "ball_association": {
                    "ball_hypotheses": {
                        "hypotheses": [hypothesis],
                    }
                }
            },
        },
    )


def test_context_projection_and_player_features() -> None:
    enricher = BallContextFeatureEnricher()
    state = _state(10, (10.0, 7.5))
    diagnostics = enricher.enrich(state)
    assert diagnostics["enabled"] is True
    assert diagnostics["decision_effect"] == "diagnostic_only"

    hypothesis = state.diagnostics["detector"]["ball_association"]["ball_hypotheses"]["hypotheses"][0]
    context = hypothesis["context"]
    assert context["homography_valid"] is True
    assert context["homography_anchor_count"] == 12
    assert context["homography_inlier_count"] == 6
    assert context["homography_inlier_ratio"] == 0.5
    assert context["homography_rmse_px"] == 1.5261
    assert context["apparent_court_xy"] == [10.0, 7.5]
    assert context["inside_court_1m_margin"] is True
    assert context["inside_player_box"] is True
    assert context["nearest_player_court_distance_m"] == 0.5
    assert context["apparent_court_speed_m_per_step"] is None


def test_context_speed_uses_same_hypothesis_only() -> None:
    enricher = BallContextFeatureEnricher()
    first = _state(20, (10.0, 7.5))
    second = _state(21, (11.0, 7.5))
    enricher.enrich(first)
    enricher.enrich(second)
    hypothesis = second.diagnostics["detector"]["ball_association"]["ball_hypotheses"]["hypotheses"][0]
    assert hypothesis["context"]["apparent_court_speed_m_per_step"] == 1.0


if __name__ == "__main__":
    test_context_projection_and_player_features()
    test_context_speed_uses_same_hypothesis_only()
    print("ball_context_features_test: PASS")
