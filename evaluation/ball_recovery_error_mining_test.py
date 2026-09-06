from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_recovery_error_mining import event_detail, replay

FRAME_SHAPE = (1000, 1000, 3)


def trace(cx: float, cy: float, confidence: float) -> dict[str, object]:
    return {
        "index": 0,
        "box": [cx - 10, cy - 10, cx + 10, cy + 10],
        "center": [cx, cy],
        "confidence": confidence,
        "association": {
            "score": confidence,
            "confidence": confidence,
            "proximity": 0.0,
            "size_similarity": 1.0,
            "distance_norm": 0.30,
            "gate_norm": 0.05,
            "plausible": False,
            "high_confidence_gate_used": False,
        },
    }


def config() -> dict[str, object]:
    return {
        "max_hold_frames": 5,
        "reset_after_frames": 12,
        "velocity_alpha": 0.40,
        "score_weights": {"confidence": 0.58, "proximity": 0.32, "size": 0.10},
        "gate": {
            "base": 0.05,
            "growth_per_frame": 0.0,
            "max": 0.05,
            "high_confidence_threshold": 0.99,
            "high_confidence_multiplier": 1.0,
        },
        "reacquire": {
            "enabled": True,
            "min_confidence": 0.45,
            "confidence_margin": 0.15,
            "confirm_frames": 2,
            "radius_norm": 0.04,
            "min_size_similarity": 0.25,
        },
    }


def production_rows() -> list[dict[str, object]]:
    cfg = config()
    return [
        {
            "frame_id": 0,
            "ball_association": {
                "source": "detector_associated",
                "selected_index": 0,
                "reacquired": False,
                "track_state": "TRACKED",
                "candidates": [trace(100, 100, 0.60)],
                "config": cfg,
            },
        },
        {
            "frame_id": 1,
            "ball_association": {
                "source": "predicted_hold",
                "selected_index": None,
                "reacquired": False,
                "track_state": "REACQUIRING",
                "challenger_streak": 1,
                "challenger_confidence": 0.35,
                "candidates": [trace(500, 500, 0.35)],
                "config": cfg,
            },
        },
        {
            "frame_id": 2,
            "ball_association": {
                "source": "detector_reacquired",
                "selected_index": 0,
                "reacquired": True,
                "track_state": "TRACKED",
                "challenger_streak": 2,
                "challenger_confidence": 0.35,
                "candidates": [trace(505, 500, 0.35)],
                "config": cfg,
            },
        },
    ]


def truth_rows() -> dict[int, dict[str, object]]:
    return {
        0: {"frame_id": 0, "visible": True, "point": [100, 100]},
        1: {"frame_id": 1, "visible": True, "point": [500, 500]},
        2: {"frame_id": 2, "visible": True, "point": [505, 500]},
    }


def test_counterfactual_replay_finds_recovery_win() -> None:
    rows = production_rows()
    truth = truth_rows()
    off, off_metrics = replay(rows, truth, FRAME_SHAPE, config(), False, 40.0, 0.25)
    on, on_metrics = replay(rows, truth, FRAME_SHAPE, config(), True, 40.0, 0.25)

    assert off[2]["selected_correct"] is False
    assert on[2]["selected_correct"] is True
    assert on[2]["reacquired"] is True
    assert on_metrics["association"]["correct_when_true_candidate_available"] > off_metrics["association"]["correct_when_true_candidate_available"]

    by_frame = {int(row["frame_id"]): row for row in rows}
    event = event_detail(2, by_frame, truth, off, on, 1, 2, 40.0, 0.25)
    assert event["production_reacquisition_correct"] is True
    assert event["counterfactual_at_event"]["label"] == "recovery_win"
    assert event["window"]["net_v2_correct_delta"] > 0


def main() -> None:
    test_counterfactual_replay_finds_recovery_win()
    print("[PASS] recovery error miner counterfactual replay")


if __name__ == "__main__":
    main()
