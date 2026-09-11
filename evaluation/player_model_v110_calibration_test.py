from __future__ import annotations

from evaluation.player_model_v110_calibration import choose_stress_threshold, evaluate_full_gate


def main() -> None:
    rows = [
        {
            "player_confidence": 0.10,
            "analytics_eligible_players_per_frame": 4.60,
            "tracker_assignment_rate_pct": 90.77,
            "untracked_observation_rate_pct": 9.23,
            "time_normalized_short_track_rate_pct": 42.99,
            "unique_track_ids_per_1000_tracked_observations": 41.23,
        },
        {
            "player_confidence": 0.20,
            "analytics_eligible_players_per_frame": 4.50,
            "tracker_assignment_rate_pct": 96.0,
            "untracked_observation_rate_pct": 4.0,
            "time_normalized_short_track_rate_pct": 30.0,
            "unique_track_ids_per_1000_tracked_observations": 30.0,
        },
        {
            "player_confidence": 0.50,
            "analytics_eligible_players_per_frame": 4.10,
            "tracker_assignment_rate_pct": 99.0,
            "untracked_observation_rate_pct": 1.0,
            "time_normalized_short_track_rate_pct": 15.0,
            "unique_track_ids_per_1000_tracked_observations": 20.0,
        },
    ]
    selected = choose_stress_threshold(rows, max_players_proxy_drop_pct=5.0)
    assert abs(selected["selected_player_confidence"] - 0.20) < 1e-9
    assert 0.50 not in selected["viable_thresholds"]

    historical = {
        "analytics_eligible_players_per_frame_macro_mean": 7.73,
        "tracker_assignment_rate_macro_pct": 98.716,
        "untracked_observation_rate_macro_pct": 1.284,
        "short_track_rate_frame_based_macro_pct": 12.392,
    }
    passing = {
        "analytics_eligible_players_per_frame_macro_mean": 7.60,
        "tracker_assignment_rate_macro_pct": 98.50,
        "untracked_observation_rate_macro_pct": 1.50,
        "short_track_rate_macro_pct": 13.50,
    }
    gate = evaluate_full_gate(passing, historical)
    assert gate["pass"] is True
    assert all(gate["checks"].values())

    failing = dict(passing)
    failing["short_track_rate_macro_pct"] = 15.0
    fail_gate = evaluate_full_gate(failing, historical)
    assert fail_gate["pass"] is False
    assert fail_gate["checks"]["short_tracks"] is False

    print("player_model_v110_calibration_test: PASS")


if __name__ == "__main__":
    main()
