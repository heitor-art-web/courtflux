from __future__ import annotations

import numpy as np

from services.vision.p1_temporal_jersey_analyzer import (
    PerformanceTemporalJerseyAnalyzer,
)


def test_uniform_sequence_indices_preserve_temporal_order() -> None:
    assert PerformanceTemporalJerseyAnalyzer._uniform_indices(8, 8) == list(range(8))
    indices = PerformanceTemporalJerseyAnalyzer._uniform_indices(17, 8)
    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == 16
    assert indices == sorted(indices)


def test_temporal_digit_distribution_recovers_23() -> None:
    tens = np.array([0.04, 0.03, 0.82, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01])
    ones = np.array([0.02, 0.03, 0.03, 0.84, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01])
    joint = PerformanceTemporalJerseyAnalyzer._joint_number_distribution(tens, ones)
    assert int(np.argmax(joint)) == 23
    assert float(joint[23]) > 0.68


def test_temporal_model_winner_is_not_rewritten_by_roster() -> None:
    analyzer = PerformanceTemporalJerseyAnalyzer.__new__(PerformanceTemporalJerseyAnalyzer)
    analyzer.rosters = {
        "Ovarense": {
            "roster": {
                "4": {"name": "Player 4"},
                "21": {"name": "Player 21"},
                "23": {"name": "Player 23"},
            }
        }
    }
    assert analyzer._valid_roster_numbers("Ovarense") == ["4", "21", "23"]

    tens = np.zeros(10, dtype=np.float64)
    ones = np.zeros(10, dtype=np.float64)
    tens[6] = 0.80
    tens[2] = 0.20
    ones[6] = 0.75
    ones[3] = 0.25
    joint = analyzer._joint_number_distribution(tens, ones)

    # Raw recognition remains #66 even though the roster contains #23.
    assert int(np.argmax(joint)) == 66
    assert "66" not in analyzer._valid_roster_numbers("Ovarense")


def test_single_digit_no_tens_bucket_stays_explicit() -> None:
    tens = np.zeros(10, dtype=np.float64)
    ones = np.zeros(10, dtype=np.float64)
    tens[0] = 0.91
    tens[2] = 0.09
    ones[4] = 0.88
    ones[1] = 0.12
    joint = PerformanceTemporalJerseyAnalyzer._joint_number_distribution(tens, ones)
    assert int(np.argmax(joint)) == 4
    assert float(tens[0]) > float(tens[2])


def main() -> None:
    test_uniform_sequence_indices_preserve_temporal_order()
    test_temporal_digit_distribution_recovers_23()
    test_temporal_model_winner_is_not_rewritten_by_roster()
    test_single_digit_no_tens_bucket_stays_explicit()
    print("p1_temporal_jersey_test: PASS")


if __name__ == "__main__":
    main()
