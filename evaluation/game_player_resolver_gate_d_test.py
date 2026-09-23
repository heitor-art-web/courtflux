from __future__ import annotations

from services.vision.game_player_resolver_gate_d import GateDGamePlayerResolverV1
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


def _resolver(**overrides) -> GateDGamePlayerResolverV1:
    values = {
        "appearance_match_similarity": 0.90,
        "appearance_plausible_similarity": 0.75,
        "appearance_match_margin": 0.05,
        "appearance_split_similarity": 0.60,
        "split_conflict_observations": 2,
        "max_track_gap_frames": 10,
        "max_association_gap_frames": 20,
    }
    values.update(overrides)
    return GateDGamePlayerResolverV1(GamePlayerResolverConfig(**values))


def _grid_descriptor(*, feature: int, column: int | None = None) -> list[float]:
    values = [0.0] * (6 * 3 * 6)
    columns = range(3) if column is None else (column,)
    for row in range(6):
        for col in columns:
            values[((row * 3 + col) * 6) + feature] = 1.0
    return values


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


def test_stale_entities_do_not_force_new_track_unknown() -> None:
    resolver = _resolver(max_association_gap_frames=5)
    original = resolver.observe(frame_id=1, track_id=1, appearance=[1.0, 0.0])
    resolver.observe(frame_id=2, track_id=1, appearance=[1.0, 0.0])

    newcomer = resolver.observe(frame_id=20, track_id=2, appearance=[1.0, 0.0])
    row = resolver.resolution(2, 20)

    assert original is not None
    assert newcomer is not None
    assert newcomer != original
    assert row is not None
    assert row["association_reason"] == "new_entity_no_compatible_history"


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
    assert row["association_reason"] == "new_entity_no_compatible_history"
    assert row["episode_open_reason"] == "initial_track_observation"


def test_horizontal_pose_shift_does_not_split_full_grid_descriptor() -> None:
    resolver = _resolver(appearance_split_similarity=0.80)
    left = _grid_descriptor(feature=0, column=0)
    right = _grid_descriptor(feature=0, column=2)

    resolver.observe(frame_id=1, track_id=5, appearance=left)
    resolver.observe(frame_id=2, track_id=5, appearance=right)
    resolver.observe(frame_id=3, track_id=5, appearance=right)

    assert len([key for key in resolver.episodes if key[0] == 5]) == 1
    row = resolver.resolution(5, 3)
    assert row is not None
    assert row["episode_index"] == 0


def test_coherent_body_change_splits_full_grid_descriptor() -> None:
    resolver = _resolver(
        appearance_split_similarity=0.80,
        split_conflict_observations=2,
    )
    first = _grid_descriptor(feature=0)
    second = _grid_descriptor(feature=1)

    resolver.observe(frame_id=1, track_id=5, appearance=first)
    resolver.observe(frame_id=2, track_id=5, appearance=second)
    resolver.observe(frame_id=3, track_id=5, appearance=second)
    row = resolver.resolution(5, 3)

    assert row is not None
    assert row["episode_index"] == 1
    assert row["episode_open_reason"] == "appearance_conflict"
    raw_similarity = row["episode_open_appearance_similarity"]
    split_similarity = row["episode_open_split_similarity"]
    candidate_similarity = row["episode_open_candidate_similarity"]
    assert raw_similarity is not None and float(raw_similarity) < 0.80
    assert split_similarity is not None and float(split_similarity) < 0.80
    assert candidate_similarity is not None and float(candidate_similarity) > 0.99


def test_incoherent_conflicts_do_not_accumulate_split_streak() -> None:
    resolver = _resolver(
        appearance_split_similarity=0.80,
        split_conflict_observations=2,
    )
    stable = _grid_descriptor(feature=0)
    conflict_a = _grid_descriptor(feature=1)
    conflict_b = _grid_descriptor(feature=2)

    resolver.observe(frame_id=1, track_id=5, appearance=stable)
    resolver.observe(frame_id=2, track_id=5, appearance=conflict_a)
    resolver.observe(frame_id=3, track_id=5, appearance=conflict_b)

    assert len([key for key in resolver.episodes if key[0] == 5]) == 1
    row = resolver.resolution(5, 3)
    assert row is not None
    assert row["episode_index"] == 0


def main() -> None:
    test_active_tracks_block_order_dependent_merge()
    test_stale_entities_do_not_force_new_track_unknown()
    test_resolution_exposes_episode_diagnostics()
    test_horizontal_pose_shift_does_not_split_full_grid_descriptor()
    test_coherent_body_change_splits_full_grid_descriptor()
    test_incoherent_conflicts_do_not_accumulate_split_streak()
    print("game_player_resolver_gate_d_test: PASS")


if __name__ == "__main__":
    main()
