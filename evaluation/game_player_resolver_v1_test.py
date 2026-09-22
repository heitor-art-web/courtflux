from __future__ import annotations

from services.vision.game_player_resolver_v1 import (
    GamePlayerResolverConfig,
    GamePlayerResolverV1,
)


def _resolver() -> GamePlayerResolverV1:
    return GamePlayerResolverV1(
        GamePlayerResolverConfig(
            appearance_match_similarity=0.90,
            appearance_plausible_similarity=0.75,
            appearance_match_margin=0.05,
            appearance_split_similarity=0.60,
            split_conflict_observations=2,
            max_track_gap_frames=10,
        )
    )


def test_cross_track_same_player_can_reuse_game_player() -> None:
    resolver = _resolver()
    first = resolver.observe(
        frame_id=10,
        track_id=1,
        appearance=[1.0, 0.0, 0.0],
        team_id="team_a",
        team_confidence=0.95,
    )
    resolver.observe(
        frame_id=11,
        track_id=1,
        appearance=[0.99, 0.03, 0.0],
        team_id="team_a",
        team_confidence=0.95,
    )
    second = resolver.observe(
        frame_id=30,
        track_id=2,
        appearance=[0.99, 0.02, 0.0],
        team_id="team_a",
        team_confidence=0.95,
    )

    assert first is not None
    assert second == first


def test_simultaneous_tracks_never_merge() -> None:
    resolver = _resolver()
    first = resolver.observe(
        frame_id=10,
        track_id=1,
        appearance=[1.0, 0.0, 0.0],
        team_id="team_a",
        team_confidence=0.95,
    )
    second = resolver.observe(
        frame_id=10,
        track_id=2,
        appearance=[1.0, 0.0, 0.0],
        team_id="team_a",
        team_confidence=0.95,
    )

    assert first is not None
    assert second is not None
    assert second != first


def test_incompatible_high_confidence_teams_never_merge() -> None:
    resolver = _resolver()
    first = resolver.observe(
        frame_id=1,
        track_id=1,
        appearance=[1.0, 0.0],
        team_id="team_a",
        team_confidence=0.95,
    )
    second = resolver.observe(
        frame_id=20,
        track_id=2,
        appearance=[1.0, 0.0],
        team_id="team_b",
        team_confidence=0.95,
    )

    assert first is not None
    assert second is not None
    assert second != first


def test_same_jersey_is_not_identity_when_appearance_disagrees() -> None:
    resolver = _resolver()
    first = resolver.observe(
        frame_id=1,
        track_id=1,
        appearance=[1.0, 0.0],
        team_id="team_a",
        team_confidence=0.95,
        jersey_number="24",
        jersey_confidence=0.95,
    )
    second = resolver.observe(
        frame_id=20,
        track_id=2,
        appearance=[0.0, 1.0],
        team_id="team_a",
        team_confidence=0.95,
        jersey_number="24",
        jersey_confidence=0.95,
    )

    assert first is not None
    # Strong visual separation may create a new entity, but same (team, jersey)
    # must never force a merge.
    assert second is not None
    assert second != first


def test_sustained_appearance_change_splits_raw_track() -> None:
    resolver = _resolver()
    first = resolver.observe(
        frame_id=1,
        track_id=7,
        appearance=[1.0, 0.0],
        team_id="team_a",
        team_confidence=0.95,
    )
    resolver.observe(
        frame_id=2,
        track_id=7,
        appearance=[0.99, 0.01],
        team_id="team_a",
        team_confidence=0.95,
    )

    # One contradictory observation does not split.
    resolver.observe(
        frame_id=3,
        track_id=7,
        appearance=[0.0, 1.0],
        team_id="team_a",
        team_confidence=0.95,
    )
    assert len([key for key in resolver.episodes if key[0] == 7]) == 1

    # A second sustained contradiction opens a new temporal episode.
    second = resolver.observe(
        frame_id=4,
        track_id=7,
        appearance=[0.0, 1.0],
        team_id="team_a",
        team_confidence=0.95,
    )

    assert len([key for key in resolver.episodes if key[0] == 7]) == 2
    assert resolver.resolve(7, 2) == first
    assert resolver.resolve(7, 4) == second
    assert second != first


def test_ambiguous_cross_track_match_abstains() -> None:
    resolver = _resolver()
    a = resolver.observe(frame_id=1, track_id=1, appearance=[1.0, 0.0])
    b = resolver.observe(frame_id=1, track_id=2, appearance=[0.98, 0.20])
    assert a is not None and b is not None and a != b

    candidate = resolver.observe(
        frame_id=20,
        track_id=3,
        appearance=[0.995, 0.10],
    )
    assert candidate is None


def test_resolve_interval_abstains_across_episode_boundary() -> None:
    resolver = _resolver()
    first = resolver.observe(frame_id=1, track_id=8, appearance=[1.0, 0.0])
    resolver.observe(frame_id=2, track_id=8, appearance=[1.0, 0.0])
    resolver.observe(frame_id=3, track_id=8, appearance=[0.0, 1.0])
    second = resolver.observe(frame_id=4, track_id=8, appearance=[0.0, 1.0])

    assert first is not None
    assert second != first
    assert resolver.resolve_interval(8, 1, 2) == first
    assert resolver.resolve_interval(8, 4, 4) == second
    assert resolver.resolve_interval(8, 2, 4) is None


def test_entity_prototype_tracks_later_episode_observations() -> None:
    resolver = _resolver()
    player = resolver.observe(frame_id=1, track_id=4, appearance=[1.0, 0.0])
    resolver.observe(frame_id=2, track_id=4, appearance=[0.8, 0.6])

    assert player is not None
    prototype = resolver.entities[player].appearance
    assert prototype is not None
    assert prototype[1] > 0.20


def main() -> None:
    test_cross_track_same_player_can_reuse_game_player()
    test_simultaneous_tracks_never_merge()
    test_incompatible_high_confidence_teams_never_merge()
    test_same_jersey_is_not_identity_when_appearance_disagrees()
    test_sustained_appearance_change_splits_raw_track()
    test_ambiguous_cross_track_match_abstains()
    test_resolve_interval_abstains_across_episode_boundary()
    test_entity_prototype_tracks_later_episode_observations()
    print("game_player_resolver_v1_test: PASS")


if __name__ == "__main__":
    main()
