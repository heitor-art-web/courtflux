from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_resolver_gate_d_v6 import GateDGamePlayerResolverV6


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
            simultaneous_track_ids={10},
        )
    assert resolver.current_episode_by_track[10] == initial


def test_jersey_input_is_ignored_and_never_splits_v6_episode() -> None:
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
            jersey_confidence=0.99,
            simultaneous_track_ids={20},
        )
    first = resolver.current_episode_by_track[20]

    for frame_id in range(8, 14):
        resolver.observe(
            frame_id=frame_id,
            track_id=20,
            appearance=appearance,
            team_id="team_a",
            team_confidence=0.95,
            jersey_number="23",
            jersey_confidence=0.99,
            simultaneous_track_ids={20},
        )

    assert resolver.current_episode_by_track[20] == first
    resolution = resolver.resolution(20, 13) or {}
    assert resolution.get("jersey_number") is None
    assert float(resolution.get("jersey_confidence", 0.0) or 0.0) == 0.0


def test_jersey_cannot_enable_long_reappearance() -> None:
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
            jersey_confidence=0.99,
            simultaneous_track_ids={30},
        )
    assert first_id is not None

    later_id = resolver.observe(
        frame_id=1000,
        track_id=31,
        appearance=(1.0, 0.0, 0.0),
        team_id="team_b",
        team_confidence=0.95,
        jersey_number="8",
        jersey_confidence=0.99,
        simultaneous_track_ids={31},
    )
    assert later_id is not None
    assert later_id != first_id


def test_simultaneity_remains_hard_reject() -> None:
    resolver = GateDGamePlayerResolverV6()
    first_id = resolver.observe(
        frame_id=1,
        track_id=40,
        appearance=(1.0, 0.0, 0.0),
        team_id="team_a",
        team_confidence=0.95,
        simultaneous_track_ids={40},
    )
    assert first_id is not None

    second_id = resolver.observe(
        frame_id=2,
        track_id=41,
        appearance=(1.0, 0.0, 0.0),
        team_id="team_a",
        team_confidence=0.95,
        simultaneous_track_ids={40, 41},
    )
    assert second_id is not None
    assert second_id != first_id


def main() -> None:
    test_team_flip_alone_does_not_split_v6_episode()
    test_jersey_input_is_ignored_and_never_splits_v6_episode()
    test_jersey_cannot_enable_long_reappearance()
    test_simultaneity_remains_hard_reject()
    print("p2_v6_gate_d_identity_test: PASS")


if __name__ == "__main__":
    main()
