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
    test_team_reversal_unlocks_old_identity()
    test_recent_jersey_evidence_can_reverse_old_number()
    print("p1_identity_v2_test: PASS")


if __name__ == "__main__":
    main()
