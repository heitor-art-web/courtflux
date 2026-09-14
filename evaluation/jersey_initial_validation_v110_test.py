from __future__ import annotations

import numpy as np

from evaluation.jersey_initial_validation_v110 import (
    _build_track_decisions,
    _summarize_decisions,
    _track_bucket,
)


def _prediction(track_id: int, number: int, uncertainty: float) -> dict:
    probs = np.full(100, 0.0001, dtype=float)
    probs[number] = 0.95
    probs[(number + 1) % 100] = 0.04
    probs /= probs.sum()
    return {
        "track_id": track_id,
        "pred_number": number,
        "pred_score": float(probs[number]),
        "top2_score": float(probs[(number + 1) % 100]),
        "number_margin": float(probs[number] - probs[(number + 1) % 100]),
        "uncertainty": uncertainty,
        "absent_argmax": False,
        "absent_margin": -1.0,
        "_number_probs": probs,
    }


def main() -> int:
    assert _track_bucket(1) == "short_1_4"
    assert _track_bucket(4) == "short_1_4"
    assert _track_bucket(5) == "medium_5_14"
    assert _track_bucket(15) == "long_15_plus"

    predictions = [
        _prediction(1, 23, 0.10),
        _prediction(1, 23, 0.20),
        _prediction(2, 7, 0.30),
    ]
    observations = {1: 20, 2: 3, 3: 2}
    decisions = _build_track_decisions(predictions, observations)
    by_id = {row["track_id"]: row for row in decisions}

    assert by_id[1]["accepted"] is True
    assert by_id[1]["prediction"] == 23
    assert by_id[1]["modal_frame_fraction"] == 1.0
    assert by_id[2]["accepted"] is False
    assert by_id[2]["unknown_reason"] == "frozen_policy_abstention"
    assert by_id[3]["accepted"] is False
    assert by_id[3]["unknown_reason"] == "no_valid_jersey_sample"

    summary = _summarize_decisions(decisions)
    assert summary["tracked_player_ids"] == 3
    assert summary["sampled_tracks"] == 2
    assert summary["policy_evaluable_tracks"] == 1
    assert summary["accepted_tracks"] == 1
    assert summary["unknown_tracks"] == 2
    assert abs(summary["operational_acceptance_rate"] - (1 / 3)) < 1e-12
    assert summary["short_track_proxy"]["tracks"] == 2
    assert summary["by_track_length"]["long_15_plus"]["accepted"] == 1

    print("jersey_initial_validation_v110_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
