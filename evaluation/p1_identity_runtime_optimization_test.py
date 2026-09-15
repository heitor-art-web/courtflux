from __future__ import annotations

from services.vision.identity_v2_detector import PerformanceDetector


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
    detector.jersey_evidence_half_life_frames = 72.0
    return detector


def test_unstable_team_profile_is_always_rescored() -> None:
    detector = _detector()
    profile = {
        "team": "FC Porto",
        "team_observations": 5,
        "team_confidence": 0.90,
        "team_margin": 0.20,
        "last_team_observation_frame": 100,
        "identity_locked": False,
    }
    assert detector._team_score_due(profile, 101) is True

    profile["team_observations"] = 6
    profile["team_confidence"] = 0.55
    assert detector._team_score_due(profile, 101) is True

    profile["team_confidence"] = 0.90
    profile["team_margin"] = 0.01
    assert detector._team_score_due(profile, 101) is True


def test_stable_profile_uses_temporal_cadence() -> None:
    detector = _detector()
    profile = {
        "team": "Ovarense",
        "team_observations": 8,
        "team_confidence": 0.75,
        "team_margin": 0.08,
        "last_team_observation_frame": 100,
        "identity_locked": False,
    }
    assert detector._team_score_due(profile, 101) is False
    assert detector._team_score_due(profile, 104) is False
    assert detector._team_score_due(profile, 105) is True


def test_locked_identity_is_still_rescored_on_cadence() -> None:
    detector = _detector()
    profile = {
        "team": "FC Porto",
        "identity_team": "FC Porto",
        "identity_locked": True,
        "team_observations": 8,
        "team_confidence": 0.80,
        "team_margin": 0.15,
        "last_team_observation_frame": 100,
        "shirt_num": "4",
        "player_name": "Player",
        "identity_revision": 0,
    }
    assert detector._team_score_due(profile, 101) is False
    assert detector._team_score_due(profile, 105) is True

    player = {}
    detector._reuse_team_profile(player, profile)
    assert player["team"] == "FC Porto"
    assert player["shirt_num"] == "4"
    assert player["identity_locked"] is True


def main() -> None:
    test_unstable_team_profile_is_always_rescored()
    test_stable_profile_uses_temporal_cadence()
    test_locked_identity_is_still_rescored_on_cadence()
    print("p1_identity_runtime_optimization_test: PASS")


if __name__ == "__main__":
    main()
