from __future__ import annotations

import numpy as np
import torch

from services.vision.p1_uncertainty_jersey_analyzer import (
    PerformanceUncertaintyJerseyAnalyzer,
    TiedDigitAwareClassifier,
)


def test_classifier_emits_100_numbers_plus_absent() -> None:
    classifier = TiedDigitAwareClassifier(embed_dim=8)
    output = classifier(torch.zeros((2, 8), dtype=torch.float32))
    assert tuple(output.shape) == (2, 101)


def test_digit_marginals_preserve_two_digit_target() -> None:
    probs = np.zeros(100, dtype=np.float64)
    probs[23] = 0.70
    probs[21] = 0.20
    probs[13] = 0.10
    tens, ones = PerformanceUncertaintyJerseyAnalyzer._digit_marginals(probs)

    assert int(np.argmax(tens)) == 2
    assert int(np.argmax(ones)) == 3
    assert abs(float(tens[2]) - 0.90) < 1e-9
    assert abs(float(ones[3]) - 0.80) < 1e-9


def test_single_digit_maps_to_diagnostic_no_tens_bucket() -> None:
    probs = np.zeros(100, dtype=np.float64)
    probs[4] = 1.0
    tens, ones = PerformanceUncertaintyJerseyAnalyzer._digit_marginals(probs)
    assert int(np.argmax(tens)) == 0
    assert int(np.argmax(ones)) == 4


def test_roster_is_validator_input_not_decoder() -> None:
    analyzer = PerformanceUncertaintyJerseyAnalyzer.__new__(
        PerformanceUncertaintyJerseyAnalyzer
    )
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

    probs = np.zeros(100, dtype=np.float64)
    probs[66] = 0.55
    probs[23] = 0.30
    probs[4] = 0.15
    top = analyzer._top_numbers(probs, limit=3)
    # The model winner remains #66 even though the roster contains #23/#4.
    assert top[0]["number"] == "66"
    assert "66" not in analyzer._valid_roster_numbers("Ovarense")


def main() -> None:
    test_classifier_emits_100_numbers_plus_absent()
    test_digit_marginals_preserve_two_digit_target()
    test_single_digit_maps_to_diagnostic_no_tens_bucket()
    test_roster_is_validator_input_not_decoder()
    print("p1_uncertainty_jersey_test: PASS")


if __name__ == "__main__":
    main()
