from __future__ import annotations

from services.vision.identity_v2_detector import PerformanceDetector


def _detector() -> PerformanceDetector:
    detector = PerformanceDetector.__new__(PerformanceDetector)
    detector.team_evidence_window_frames = 120
    detector.team_evidence_half_life_frames = 45.0
    detector.team_reversal_min_observations = 3
    return detector


def _profile() -> dict:
    return {
        "team": "team_a",
        "team_evidence": [
            {"frame_id": frame, "scores": {"team_a": 0.90, "team_b": 0.10}, "weight": 1.0}
            for frame in (40, 50, 60, 70, 80, 90)
        ],
        "team_corrections": 0,
        "team_reversal_candidate": None,
        "team_reversal_streak": 0,
        "identity_revision": 0,
        "identity_locked": False,
        "jersey_evidence": [],
    }


def test_three_consecutive_strong_fresh_rescores_reverse_stale_profile() -> None:
    detector = _detector()
    profile = _profile()
    confirmed = False

    for frame in (100, 105, 110):
        profile["team_evidence"].append(
            {"frame_id": frame, "scores": {"team_a": 0.10, "team_b": 0.90}, "weight": 1.0}
        )
        confirmed = detector._observe_team_reversal_candidate(
            profile,
            candidate="team_b",
            confidence=0.90,
            margin=0.80,
            accepted=True,
        )
        if confirmed:
            profile["team_evidence"] = profile["team_evidence"][-detector.team_reversal_min_observations :]

    assert confirmed is True
    assert len(profile["team_evidence"]) == 3

    best, confidence, margin, observations = detector._aggregate_team_evidence(profile, 110)
    detector._apply_team_resolution(
        profile,
        best_team=best,
        confidence=confidence,
        margin=margin,
        observations=observations,
    )

    assert best == "team_b"
    assert observations == 3
    assert confidence >= 0.64
    assert margin >= 0.18
    assert profile["team"] == "team_b"
    assert profile["team_corrections"] == 1


def test_weak_or_rejected_rescore_breaks_reversal_streak() -> None:
    detector = _detector()
    profile = _profile()

    assert detector._observe_team_reversal_candidate(
        profile,
        candidate="team_b",
        confidence=0.90,
        margin=0.80,
        accepted=True,
    ) is False
    assert profile["team_reversal_streak"] == 1

    assert detector._observe_team_reversal_candidate(
        profile,
        candidate="team_b",
        confidence=0.60,
        margin=0.20,
        accepted=True,
    ) is False
    assert profile["team_reversal_candidate"] is None
    assert profile["team_reversal_streak"] == 0

    assert detector._observe_team_reversal_candidate(
        profile,
        candidate="team_b",
        confidence=0.90,
        margin=0.80,
        accepted=False,
    ) is False
    assert profile["team_reversal_streak"] == 0


def main() -> None:
    test_three_consecutive_strong_fresh_rescores_reverse_stale_profile()
    test_weak_or_rejected_rescore_breaks_reversal_streak()
    print("p2_team_reversal_temporal_test: PASS")


if __name__ == "__main__":
    main()
