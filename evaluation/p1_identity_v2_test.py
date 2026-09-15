from __future__ import annotations

import numpy as np

from services.vision.identity_v2_detector import PerformanceDetector
from services.vision.p1_jersey_analyzer import PerformanceJerseyAnalyzer


def _detector() -> PerformanceDetector:
    detector = PerformanceDetector.__new__(PerformanceDetector)
    detector.player_db = {}
    detector.team_stable_min_observations = 6
    detector.team_stable_cadence_frames = 5
    detector.team_stable_confidence = 0.60
    detector.team_stable_margin = 0.02
    detector.team_evidence_window_frames = 120
    detector.team_evidence_half_life_frames = 45.0
    detector.team_reversal_min_observations = 3
    detector.jersey_evidence_window_frames = 180
    detector.jersey_evidence_half_life_frames = 36.0
    return detector


def _digit_row(
    *,
    tens: dict[int, float],
    ones: dict[int, float],
    quality: float = 0.8,
) -> dict:
    tens_probs = [0.0] * 10
    ones_probs = [0.0] * 10
    for digit, value in tens.items():
        tens_probs[int(digit)] = float(value)
    for digit, value in ones.items():
        ones_probs[int(digit)] = float(value)
    tens_total = sum(tens_probs) or 1.0
    ones_total = sum(ones_probs) or 1.0
    return {
        "tens_probs": [value / tens_total for value in tens_probs],
        "ones_probs": [value / ones_total for value in ones_probs],
        "crop_quality": quality,
    }


def test_best_crop_buffer_keeps_quality_not_just_recency() -> None:
    analyzer = PerformanceJerseyAnalyzer.__new__(PerformanceJerseyAnalyzer)
    analyzer.sequence_length = 2
    analyzer.track_crop_buffers = {}
    analyzer.track_observation_index = {}
    analyzer.best_crop_age_penalty = 0.006

    poor = np.full((120, 70, 3), 110, dtype=np.uint8)
    sharp = np.zeros((120, 70, 3), dtype=np.uint8)
    sharp[:, ::2] = 255

    analyzer.observe_track(poor, 7, frame_id=1)
    analyzer.observe_track(poor, 7, frame_id=2)
    before = max(float(item["quality"]) for item in analyzer.track_crop_buffers[7])
    analyzer.observe_track(sharp, 7, frame_id=3)
    after = max(float(item["quality"]) for item in analyzer.track_crop_buffers[7])

    assert len(analyzer.track_crop_buffers[7]) == 2
    assert after > before


def test_raw_temporal_consensus_beats_single_outlier() -> None:
    rows = [
        {"raw_canonical": "4", "raw_probability": 0.60, "crop_quality": 0.8},
        {"raw_canonical": "4", "raw_probability": 0.52, "crop_quality": 0.8},
        {"raw_canonical": "6", "raw_probability": 0.90, "crop_quality": 0.8},
    ]
    result = PerformanceJerseyAnalyzer._raw_consensus(rows)
    assert result["candidate_number"] == "4"
    assert result["candidate_count"] == 2
    assert result["candidate_posterior"] > 0.50


def test_digit_probability_consensus_recovers_two_digit_candidate() -> None:
    rows = [
        _digit_row(tens={2: 0.84, 0: 0.10, 1: 0.06}, ones={1: 0.82, 4: 0.12, 3: 0.06}),
        _digit_row(tens={2: 0.78, 0: 0.15, 1: 0.07}, ones={1: 0.86, 4: 0.08, 3: 0.06}),
        _digit_row(tens={2: 0.81, 0: 0.12, 1: 0.07}, ones={1: 0.79, 4: 0.14, 3: 0.07}),
    ]
    result = PerformanceJerseyAnalyzer._probabilistic_digit_consensus(rows)

    assert result["candidate_number"] == "21"
    assert result["candidate_count"] == 3
    assert result["candidate_posterior"] > 0.60
    assert result["single_digit_gate"]["required"] is False


def test_ambiguous_tens_rejects_single_digit_collapse() -> None:
    # The raw top pair is 04, but tens=2 remains plausible. Identity must abstain
    # rather than converting an ambiguous two-digit jersey into #4.
    rows = [
        _digit_row(tens={0: 0.55, 2: 0.39, 1: 0.06}, ones={4: 0.75, 1: 0.15, 3: 0.10}),
        _digit_row(tens={0: 0.58, 2: 0.35, 1: 0.07}, ones={4: 0.72, 1: 0.18, 3: 0.10}),
        _digit_row(tens={0: 0.57, 2: 0.37, 1: 0.06}, ones={4: 0.78, 1: 0.12, 3: 0.10}),
    ]
    result = PerformanceJerseyAnalyzer._probabilistic_digit_consensus(rows)

    assert result["candidate_number"] == "4"
    assert result["single_digit_gate"]["required"] is True
    assert result["single_digit_gate"]["passed"] is False
    assert result["single_digit_gate"]["reason"] == "ambiguous_single_digit_tens"
    assert result["single_digit_gate"]["strongest_non_single_digit"] == 2


def test_strong_single_digit_evidence_still_allows_number() -> None:
    rows = [
        _digit_row(tens={0: 0.93, 2: 0.04, 1: 0.03}, ones={4: 0.88, 1: 0.07, 3: 0.05}),
        _digit_row(tens={0: 0.90, 2: 0.06, 1: 0.04}, ones={4: 0.91, 1: 0.05, 3: 0.04}),
        _digit_row(tens={0: 0.92, 2: 0.05, 1: 0.03}, ones={4: 0.89, 1: 0.06, 3: 0.05}),
    ]
    result = PerformanceJerseyAnalyzer._probabilistic_digit_consensus(rows)

    assert result["candidate_number"] == "4"
    assert result["candidate_count"] == 3
    assert result["single_digit_gate"]["required"] is True
    assert result["single_digit_gate"]["passed"] is True
    assert result["candidate_posterior"] > 0.80


def test_recalibrated_runtime_scale_single_digit_can_pass() -> None:
    # Runtime v3 joint pair scores are much smaller than the old vote posterior.
    # A stable one-digit candidate around the observed strong tail should pass,
    # but only with strong no-tens evidence across almost all crops.
    rows = [
        _digit_row(
            tens={0: 0.68, 6: 0.12, 2: 0.08, 4: 0.07, 1: 0.05},
            ones={4: 0.46, 6: 0.20, 8: 0.14, 9: 0.10, 1: 0.10},
            quality=0.95,
        )
        for _ in range(8)
    ]
    result = PerformanceJerseyAnalyzer._probabilistic_digit_consensus(rows)

    assert result["candidate_number"] == "4"
    assert result["candidate_count"] == 8
    assert result["single_digit_gate"]["passed"] is True
    accepted, reason = PerformanceJerseyAnalyzer._acceptance_decision(
        candidate="4",
        support=result["candidate_count"],
        single_digit_gate=result["single_digit_gate"],
        posterior=result["candidate_posterior"],
        margin=result["candidate_margin"],
        team_name="Ovarense",
        valid_roster_numbers=["4", "7", "10", "11", "12", "13", "21", "22", "23", "41"],
        roster_valid=True,
    )
    assert accepted is True
    assert reason == "accepted"


def test_recalibrated_gate_keeps_ambiguous_single_digit_unknown() -> None:
    accepted, reason = PerformanceJerseyAnalyzer._acceptance_decision(
        candidate="4",
        support=8,
        single_digit_gate={"required": True, "passed": False},
        posterior=0.24,
        margin=0.08,
        team_name="Ovarense",
        valid_roster_numbers=["4", "21", "23"],
        roster_valid=True,
    )
    assert accepted is False
    assert reason == "ambiguous_single_digit_tens"


def test_recalibrated_gate_accepts_supported_two_digit_runtime_score() -> None:
    accepted, reason = PerformanceJerseyAnalyzer._acceptance_decision(
        candidate="21",
        support=3,
        single_digit_gate={"required": False, "passed": True},
        posterior=0.20,
        margin=0.04,
        team_name="Ovarense",
        valid_roster_numbers=["4", "21", "23"],
        roster_valid=True,
    )
    assert accepted is True
    assert reason == "accepted"


def test_team_reversal_unlocks_old_identity() -> None:
    detector = _detector()
    profile = detector._profile(9)
    profile.update(
        {
            "team": "FC Porto",
            "identity_team": "FC Porto",
            "identity_locked": True,
            "shirt_num": "4",
            "player_name": "Old player",
        }
    )

    detector._apply_team_resolution(
        profile,
        best_team="Ovarense",
        confidence=0.82,
        margin=0.64,
        observations=4,
    )

    assert profile["team"] == "Ovarense"
    assert profile["identity_locked"] is False
    assert profile["shirt_num"] is None
    assert profile["player_name"] is None
    assert profile["team_corrections"] == 1
    assert profile["identity_revision"] == 1


def test_recent_jersey_evidence_can_reverse_old_number() -> None:
    detector = _detector()
    profile = detector._profile(12)
    profile["jersey_evidence"] = [
        {"frame_id": 0, "number": "4", "weight": 1.0},
        {"frame_id": 100, "number": "6", "weight": 0.7},
        {"frame_id": 112, "number": "6", "weight": 0.7},
    ]

    number, share, margin, count, _ = detector._aggregate_jersey_evidence(profile, 112)
    assert number == "6"
    assert count == 2
    assert share > 0.70
    assert margin > 0.40


def main() -> None:
    test_best_crop_buffer_keeps_quality_not_just_recency()
    test_raw_temporal_consensus_beats_single_outlier()
    test_digit_probability_consensus_recovers_two_digit_candidate()
    test_ambiguous_tens_rejects_single_digit_collapse()
    test_strong_single_digit_evidence_still_allows_number()
    test_recalibrated_runtime_scale_single_digit_can_pass()
    test_recalibrated_gate_keeps_ambiguous_single_digit_unknown()
    test_recalibrated_gate_accepts_supported_two_digit_runtime_score()
    test_team_reversal_unlocks_old_identity()
    test_recent_jersey_evidence_can_reverse_old_number()
    print("p1_identity_v2_test: PASS")


if __name__ == "__main__":
    main()
