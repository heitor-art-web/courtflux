from __future__ import annotations

from services.vision.game_player_resolver_gate_d_v2 import GateDGamePlayerResolverV2
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


def _resolver(**overrides) -> GateDGamePlayerResolverV2:
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
    return GateDGamePlayerResolverV2(GamePlayerResolverConfig(**values))


def _descriptor(columns: tuple[int | None, int | None, int | None]) -> list[float]:
    values = [0.0] * (6 * 3 * 6)
    for row in range(6):
        for col, feature in enumerate(columns):
            if feature is not None:
                values[((row * 3 + col) * 6) + feature] = 1.0
    return values


def _observe_frame(
    resolver: GateDGamePlayerResolverV2,
    frame_id: int,
    appearances: dict[int, list[float]],
) -> dict[str, object]:
    track_ids = sorted(appearances)
    context = resolver.prepare_frame(
        frame_id=frame_id,
        observations=[(track_id, appearances[track_id]) for track_id in track_ids],
    )
    for track_id in track_ids:
        resolver.observe(
            frame_id=frame_id,
            track_id=track_id,
            appearance=appearances[track_id],
            simultaneous_track_ids=track_ids,
        )
    return context


def _seed_four_tracks(resolver: GateDGamePlayerResolverV2) -> list[float]:
    stable = _descriptor((0, 0, None))
    for frame_id in range(1, 7):
        _observe_frame(
            resolver,
            frame_id,
            {1: stable, 2: stable, 3: stable, 4: stable},
        )
    return stable


def test_isolated_raw_plus_coarse_change_can_split() -> None:
    resolver = _resolver()
    stable = _seed_four_tracks(resolver)
    switched = _descriptor((0, 1, None))

    context = _observe_frame(
        resolver,
        7,
        {1: switched, 2: stable, 3: stable, 4: stable},
    )
    assert context["global_visual_shift"] is False
    _observe_frame(
        resolver,
        8,
        {1: switched, 2: stable, 3: stable, 4: stable},
    )

    row = resolver.resolution(1, 8)
    assert row is not None
    assert row["episode_index"] == 1
    assert row["episode_open_reason"] == "appearance_conflict_raw"
    assert len([key for key in resolver.episodes if key[0] == 1]) == 2


def test_horizontal_pose_shift_stays_one_episode() -> None:
    resolver = _resolver()
    left = _descriptor((0, None, None))
    right = _descriptor((None, None, 0))

    resolver.prepare_frame(frame_id=1, observations=[(1, left)])
    resolver.observe(frame_id=1, track_id=1, appearance=left)
    resolver.prepare_frame(frame_id=2, observations=[(1, right)])
    resolver.observe(frame_id=2, track_id=1, appearance=right)
    resolver.prepare_frame(frame_id=3, observations=[(1, right)])
    resolver.observe(frame_id=3, track_id=1, appearance=right)

    assert len([key for key in resolver.episodes if key[0] == 1]) == 1


def test_global_visual_shift_rebases_without_fragmenting_tracks() -> None:
    resolver = _resolver()
    _seed_four_tracks(resolver)
    shifted = _descriptor((1, 1, None))

    context = _observe_frame(
        resolver,
        7,
        {1: shifted, 2: shifted, 3: shifted, 4: shifted},
    )
    assert context["global_visual_shift"] is True
    assert context["comparable_tracks"] == 4
    assert context["conflicting_tracks"] == 4

    second_context = _observe_frame(
        resolver,
        8,
        {1: shifted, 2: shifted, 3: shifted, 4: shifted},
    )
    assert second_context["global_visual_shift"] is False
    for track_id in (1, 2, 3, 4):
        assert len([key for key in resolver.episodes if key[0] == track_id]) == 1


def main() -> None:
    test_isolated_raw_plus_coarse_change_can_split()
    test_horizontal_pose_shift_stays_one_episode()
    test_global_visual_shift_rebases_without_fragmenting_tracks()
    print("game_player_resolver_gate_d_v2_test: PASS")


if __name__ == "__main__":
    main()
