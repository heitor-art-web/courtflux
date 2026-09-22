from __future__ import annotations

from services.vision.game_player_resolver_gate_d import GateDGamePlayerResolverV1
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


def _resolver() -> GateDGamePlayerResolverV1:
    return GateDGamePlayerResolverV1(
        GamePlayerResolverConfig(
            appearance_match_similarity=0.90,
            appearance_plausible_similarity=0.75,
            appearance_match_margin=0.05,
            appearance_split_similarity=0.60,
            split_conflict_observations=2,
            max_track_gap_frames=10,
        )
    )


def test_active_tracks_block_order_dependent_merge() -> None:
    resolver = _resolver()
    original = resolver.observe(frame_id=1, track_id=10, appearance=[1.0, 0.0])
    resolver.observe(frame_id=2, track_id=10, appearance=[1.0, 0.0])

    newcomer = resolver.observe(
        frame_id=3,
        track_id=20,
        appearance=[1.0, 0.0],
        simultaneous_track_ids=[10, 20],
    )
    incumbent = resolver.observe(
        frame_id=3,
        track_id=10,
        appearance=[1.0, 0.0],
        simultaneous_track_ids=[10, 20],
    )

    assert original is not None
    assert newcomer is not None
    assert incumbent is not None
    assert newcomer != incumbent


def test_resolution_exposes_episode_diagnostics() -> None:
    resolver = _resolver()
    player = resolver.observe(
        frame_id=12,
        track_id=5,
        appearance=[1.0, 0.0],
        team_id="team_a",
        team_confidence=0.9,
        jersey_number="24",
        jersey_confidence=0.7,
    )
    row = resolver.resolution(5, 12)

    assert row is not None
    assert row["game_player_id"] == player
    assert row["episode_index"] == 0
    assert row["team_id"] == "team_a"
    assert row["jersey_number"] == "24"


def main() -> None:
    test_active_tracks_block_order_dependent_merge()
    test_resolution_exposes_episode_diagnostics()
    print("game_player_resolver_gate_d_test: PASS")


if __name__ == "__main__":
    main()
