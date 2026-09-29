from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_resolver_gate_d_v6 import GateDGamePlayerResolverV6
from services.vision.gate_d_rosterless_jersey_v6 import (
    GateDRosterlessJerseyEvidenceV6,
)


def _ocr(
    track_id: int,
    number: str,
    *,
    posterior: float = 0.92,
    quality: float = 0.75,
) -> dict[str, object]:
    return {
        "track_id": track_id,
        "accepted": False,
        "rejection_reason": "no_known_roster_numbers",
        "candidate_number": number,
        "candidate_posterior": posterior,
        "mean_crop_quality": quality,
    }


def test_rosterless_jersey_requires_repeated_visual_agreement() -> None:
    evidence = GateDRosterlessJerseyEvidenceV6()
    for frame_id in (10, 34):
        evidence.observe_predictions(
            frame_id=frame_id,
            predictions=[_ocr(7, "23")],
        )
    number, confidence, diag = evidence.evidence(track_id=7, frame_id=34)
    assert number is None
    assert confidence == 0.0
    assert diag["winner_count"] == 2

    evidence.observe_predictions(frame_id=58, predictions=[_ocr(7, "23")])
    number, confidence, diag = evidence.evidence(track_id=7, frame_id=58)
    assert number == "23"
    assert confidence >= 0.70
    assert diag["winner_count"] == 3

    evidence.observe_predictions(frame_id=82, predictions=[_ocr(7, "7")])
    number, _confidence, _diag = evidence.evidence(track_id=7, frame_id=82)
    assert number == "23"


def test_team_flip_alone_does_not_split_v6_episode() -> None:
    resolver = GateDGamePlayerResolverV6()
    appearance = (1.0, 0.0, 0.0)
    for frame_id in range(1, 6):
        resolver.observe(
            frame_id=frame_id,
            track_id=10,
            appearance=appearance,
            team_id="team_a",
            team_confidence=0.98,
            jersey_number=None,
            jersey_confidence=0.0,
            simultaneous_track_ids={10},
        )
    initial = resolver.current_episode_by_track[10]

    for frame_id in (6, 7, 8):
        resolver.observe(
            frame_id=frame_id,
            track_id=10,
            appearance=appearance,
            team_id="team_b",
            team_confidence=0.99,
            jersey_number=None,
            jersey_confidence=0.0,
            simultaneous_track_ids={10},
        )
    assert resolver.current_episode_by_track[10] == initial


def test_sustained_soft_jersey_change_splits_raw_track() -> None:
    resolver = GateDGamePlayerResolverV6()
    appearance = (1.0, 0.0, 0.0)
    for frame_id in range(1, 8):
        resolver.observe(
            frame_id=frame_id,
            track_id=20,
            appearance=appearance,
            team_id="team_a",
            team_confidence=0.95,
            jersey_number="7",
            jersey_confidence=0.90,
            simultaneous_track_ids={20},
        )
    first = resolver.current_episode_by_track[20]

    resolver.observe(
        frame_id=8,
        track_id=20,
        appearance=appearance,
        team_id="team_a",
        team_confidence=0.95,
        jersey_number="23",
        jersey_confidence=0.90,
        simultaneous_track_ids={20},
    )
    assert resolver.current_episode_by_track[20] == first

    resolver.observe(
        frame_id=9,
        track_id=20,
        appearance=appearance,
        team_id="team_a",
        team_confidence=0.95,
        jersey_number="23",
        jersey_confidence=0.90,
        simultaneous_track_ids={20},
    )
    second = resolver.current_episode_by_track[20]
    assert second != first
    assert second[1] == first[1] + 1
    resolution = resolver.resolution(20, 9) or {}
    assert resolution.get("episode_open_reason") == "jersey_conflict_soft_v6"
    assert resolution.get("jersey_number") == "23"


def test_long_reappearance_can_reuse_entity_only_with_stable_matching_jersey() -> None:
    resolver = GateDGamePlayerResolverV6()
    first_id = None
    for frame_id in range(1, 21):
        first_id = resolver.observe(
            frame_id=frame_id,
            track_id=30,
            appearance=(1.0, 0.0, 0.0),
            team_id="team_b",
            team_confidence=0.95,
            jersey_number="8",
            jersey_confidence=0.90,
            simultaneous_track_ids={30},
        )
    assert first_id is not None

    recovered = None
    for offset in range(24):
        recovered = resolver.observe(
            frame_id=1000 + offset,
            track_id=31,
            appearance=(0.8, 0.6, 0.0),
            team_id="team_b",
            team_confidence=0.95,
            jersey_number="8",
            jersey_confidence=0.90,
            simultaneous_track_ids={31},
        )
        if recovered is not None:
            break
    assert recovered == first_id

    different_id = None
    for offset in range(4):
        different_id = resolver.observe(
            frame_id=2000 + offset,
            track_id=32,
            appearance=(0.8, 0.6, 0.0),
            team_id="team_b",
            team_confidence=0.95,
            jersey_number="9",
            jersey_confidence=0.90,
            simultaneous_track_ids={32},
        )
    assert different_id is not None
    assert different_id != first_id


def main() -> None:
    test_rosterless_jersey_requires_repeated_visual_agreement()
    test_team_flip_alone_does_not_split_v6_episode()
    test_sustained_soft_jersey_change_splits_raw_track()
    test_long_reappearance_can_reuse_entity_only_with_stable_matching_jersey()
    print("p2_v6_gate_d_identity_test: PASS")


if __name__ == "__main__":
    main()
