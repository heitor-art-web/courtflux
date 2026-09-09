from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.ball_hypothesis_tracker import (
    BallIdentitySeedModel,
    IDENTITY_FEATURES,
    allows_concrete_output,
    feature_vector,
    lifecycle_state,
)


def test_lifecycle_contract() -> None:
    assert lifecycle_state(tracker_id=-1, time_since_update=0) == "TENTATIVE"
    assert lifecycle_state(tracker_id=7, time_since_update=0) == "TRACKED"
    assert lifecycle_state(tracker_id=7, time_since_update=1) == "LOST"


def test_lost_and_tentative_never_emit_concrete_output() -> None:
    assert not allows_concrete_output("TENTATIVE", observed=True, identity_seed=True)
    assert not allows_concrete_output("LOST", observed=False, identity_seed=True)
    assert not allows_concrete_output("REMOVED", observed=False, identity_seed=True)
    assert not allows_concrete_output("TRACKED", observed=True, identity_seed=False)
    assert allows_concrete_output("TRACKED", observed=True, identity_seed=True)


def _features() -> dict[str, float]:
    return {
        "detector_confidence": 0.8,
        "rank_score": 0.7,
        "confidence_rank_in_confirmed": 1.0,
        "confirmed_competitors": 2.0,
        "track_observations": 4.0,
        "track_streak": 3.0,
        "track_mean_confidence": 0.65,
        "track_area_cv": 0.2,
        "track_speed_px_per_step": 12.0,
        "box_area": 700.0,
    }


def test_feature_transform_matches_frozen_schema() -> None:
    vector = feature_vector(_features())
    assert vector.shape == (len(IDENTITY_FEATURES),)
    assert np.isfinite(vector).all()
    assert vector[0] == 0.8
    assert vector[1] == 0.7
    assert vector[4] == np.log1p(4.0)
    assert vector[8] == np.log1p(12.0)


def test_identity_model_load_and_score() -> None:
    payload = {
        "version": "test",
        "features": IDENTITY_FEATURES,
        "mean": [0.0] * len(IDENTITY_FEATURES),
        "scale": [1.0] * len(IDENTITY_FEATURES),
        "weights": [0.0] + [0.1] * len(IDENTITY_FEATURES),
        "threshold": 0.5,
        "seed_precision_target": 0.9,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = BallIdentitySeedModel.load(path)
        score = model.score(_features())
        assert 0.5 < score < 1.0
        assert model.diagnostics()["version"] == "test"


def test_identity_model_rejects_schema_drift() -> None:
    payload = {
        "version": "bad",
        "features": list(reversed(IDENTITY_FEATURES)),
        "mean": [0.0] * len(IDENTITY_FEATURES),
        "scale": [1.0] * len(IDENTITY_FEATURES),
        "weights": [0.0] + [0.1] * len(IDENTITY_FEATURES),
        "threshold": 0.5,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            BallIdentitySeedModel.load(path)
        except ValueError:
            pass
        else:
            raise AssertionError("expected schema mismatch to fail")


def main() -> None:
    test_lifecycle_contract()
    test_lost_and_tentative_never_emit_concrete_output()
    test_feature_transform_matches_frozen_schema()
    test_identity_model_load_and_score()
    test_identity_model_rejects_schema_drift()
    print("ball_hypothesis_tracker_test: PASS")


if __name__ == "__main__":
    main()
