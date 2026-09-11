from __future__ import annotations

from core.ball_state import BallStateStatus
from core.config import settings
from services.vision.ball_state_v1 import BallStateEstimatorV1


def candidate(confidence: float, center: tuple[float, float]) -> dict:
    cx, cy = center
    return {
        "class_name": "basketball",
        "confidence": confidence,
        "center": [cx, cy],
        "box": [int(cx - 5), int(cy - 5), int(cx + 5), int(cy + 5)],
    }


def main() -> None:
    assert settings.model == "weights/model_v1.1.0.pt"
    assert settings.detector_version == "1.1.0"
    assert settings.ball_candidate_confidence == 0.075
    assert settings.ball_max_prediction_gap == 4
    assert settings.ball_trajectory_min_score == 0.65

    estimator = BallStateEstimatorV1()
    frame_shape = (720, 1280, 3)

    observed = estimator.update(
        [candidate(0.90, (100.0, 100.0))],
        frame_id=0,
        timestamp_sec=0.0,
        frame_shape=frame_shape,
    )
    assert observed.status is BallStateStatus.OBSERVED
    assert observed.visible is True
    assert observed.available is True
    assert observed.predicted is False
    assert observed.position() == (100.0, 100.0)
    assert observed.confidence == 0.90
    assert abs(observed.uncertainty - 0.10) < 1e-9
    assert observed.source == "detector_seed"
    assert observed.provenance["detector_version"] == "1.1.0"
    assert observed.provenance["direct_observation"] is True
    assert observed.provenance["prediction_updates_state"] is False

    predicted = estimator.update(
        [],
        frame_id=1,
        timestamp_sec=1.0 / 25.0,
        frame_shape=frame_shape,
    )
    assert predicted.status is BallStateStatus.PREDICTED
    assert predicted.visible is False
    assert predicted.available is True
    assert predicted.predicted is True
    assert predicted.position() is not None
    assert predicted.source == "predicted_short_gap"
    assert predicted.age_since_detection == 1
    assert predicted.confidence < observed.confidence
    assert predicted.uncertainty > observed.uncertainty
    assert predicted.provenance["short_gap_prediction"] is True

    missing = estimator.update(
        [],
        frame_id=5,
        timestamp_sec=5.0 / 25.0,
        frame_shape=frame_shape,
    )
    assert missing.status is BallStateStatus.MISSING
    assert missing.visible is False
    assert missing.available is False
    assert missing.position() is None
    assert missing.box is None
    assert missing.confidence == 0.0
    assert missing.uncertainty == 1.0
    assert missing.source == "missing"

    reseeded = estimator.update(
        [candidate(0.88, (120.0, 105.0))],
        frame_id=6,
        timestamp_sec=6.0 / 25.0,
        frame_shape=frame_shape,
    )
    assert reseeded.status is BallStateStatus.OBSERVED
    assert reseeded.source == "detector_seed"

    filtered = BallStateEstimatorV1().update(
        [candidate(0.05, (50.0, 50.0))],
        frame_id=0,
        timestamp_sec=0.0,
        frame_shape=frame_shape,
    )
    assert filtered.status is BallStateStatus.MISSING

    metadata = estimator.metadata()
    assert metadata["version"] == "ball_state_v1"
    assert metadata["p2_event_semantics"] is False
    assert metadata["states"] == ["observed", "predicted", "missing"]

    print("ball_state_v1_test: PASS")


if __name__ == "__main__":
    main()
