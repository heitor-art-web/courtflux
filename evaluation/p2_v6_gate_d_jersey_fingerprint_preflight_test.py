from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_v6_gate_d_jersey_fingerprint_preflight as probe


def main() -> None:
    assert abs(probe._bc(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1.0) < 1e-9
    assert abs(probe._bc(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9
    assert abs(probe._cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) - 1.0) < 1e-9

    gt = {
        "same_pairs": [
            {"id": "S1", "left": "A1", "right": "A2", "time_gap_sec": 10},
            {"id": "S2", "left": "B1", "right": "B2", "time_gap_sec": 10},
        ],
        "different_pairs": [
            {"id": "D1", "left": "A1", "right": "B1", "time_gap_sec": 1},
            {"id": "D2", "left": "A2", "right": "B2", "time_gap_sec": 1},
        ],
    }
    fingerprints = {
        "A1": {
            "tens": np.array([0.99, 0.01]),
            "ones": np.array([0.98, 0.02]),
            "projection": np.array([1.0, 0.0]),
        },
        "A2": {
            "tens": np.array([0.98, 0.02]),
            "ones": np.array([0.97, 0.03]),
            "projection": np.array([0.99, 0.01]),
        },
        "B1": {
            "tens": np.array([0.01, 0.99]),
            "ones": np.array([0.02, 0.98]),
            "projection": np.array([0.0, 1.0]),
        },
        "B2": {
            "tens": np.array([0.02, 0.98]),
            "ones": np.array([0.03, 0.97]),
            "projection": np.array([0.01, 0.99]),
        },
    }

    for representation in ("digit_posterior", "projection", "equal_fusion"):
        result = probe._score_representation(gt, fingerprints, representation)
        selected = result["selected_operating_point"]
        assert selected["same_correct"] == 2, (representation, selected)
        assert selected["different_false_merges"] == 0, (representation, selected)
        assert result["similarity"]["pairwise_auc"] == 1.0, (representation, result)

    row = {
        "team_predictions": [
            {"track_id": 7, "box": [1, 2, 3, 4]},
        ],
        "game_player_predictions": [
            {"track_id": 7, "box": [5, 6, 7, 8]},
        ],
    }
    pred = probe._raw_track_prediction(row, 7)
    assert pred is not None
    assert pred["box"] == [1, 2, 3, 4]
    assert pred["reid_crop_source"] == "team_predictions"

    print("p2_v6_gate_d_jersey_fingerprint_preflight_test: PASS")


if __name__ == "__main__":
    main()
