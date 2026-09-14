from __future__ import annotations

from evaluation.jersey_model_comparison_v110 import compare


def _runtime() -> dict:
    return {
        "video": "/tmp/game.mp4",
        "detector": "/tmp/model_v1.1.0.pt",
        "tracker": "botsort.yaml",
        "imgsz": 960,
        "tracker_input_confidence": 0.40,
        "start_sec": 0.0,
        "end_sec": 120.0,
    }


def main() -> int:
    temporal = {
        "runtime": {**_runtime(), "jersey_model": "/tmp/best_temporal.pt"},
        "tracking": {"tracked_player_ids": 10, "short_track_proxy_rate": 0.2},
        "jersey_operational": {
            "predicted_tracks": 7,
            "unknown_tracks": 3,
            "prediction_coverage": 0.7,
            "full_support_coverage": 0.3,
            "multi_sequence_stability_rate": 0.8,
            "median_modal_fraction": 1.0,
            "prediction_distribution": [["8", 5], ["9", 2]],
        },
    }
    uncertainty = {
        "runtime": {**_runtime(), "jersey_model": "/tmp/jersey_uncertainty_vit_b.pt"},
        "jersey": {
            "tracked_player_ids": 10,
            "accepted_tracks": 4,
            "unknown_tracks": 6,
            "operational_acceptance_rate": 0.4,
            "policy_evaluable_tracks": 6,
            "short_track_proxy": {"rate": 0.2},
            "accepted_track_stability": {"median_modal_frame_fraction": 0.75},
            "accepted_number_distribution": {"12": 2, "7": 1, "23": 1},
        },
    }

    report = compare(temporal, uncertainty)
    assert report["recognizers"]["best_temporal"]["operational_coverage"] == 0.7
    assert report["recognizers"]["vit_b_uncertainty"]["operational_coverage"] == 0.4
    assert report["recognizers"]["vit_b_uncertainty"]["prediction_distribution"][0] == ["12", 2]

    bad = dict(uncertainty)
    bad["runtime"] = dict(uncertainty["runtime"])
    bad["runtime"]["tracker_input_confidence"] = 0.10
    try:
        compare(temporal, bad)
    except ValueError as exc:
        assert "runtime contracts differ" in str(exc)
    else:
        raise AssertionError("runtime mismatch must be rejected")

    print("jersey_model_comparison_v110_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
