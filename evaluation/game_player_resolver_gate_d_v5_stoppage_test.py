from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_resolver_gate_d_v5_stoppage import (
    GateDStoppageAwareGamePlayerResolverV5,
)
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


APPEARANCE = [1.0, 0.0]


def _resolver() -> GateDStoppageAwareGamePlayerResolverV5:
    return GateDStoppageAwareGamePlayerResolverV5(
        GamePlayerResolverConfig(
            team_hard_confidence=0.70,
            appearance_match_similarity=0.88,
            appearance_plausible_similarity=0.78,
            appearance_match_margin=0.06,
            appearance_split_similarity=0.58,
            split_conflict_observations=2,
            max_track_gap_frames=24,
            max_association_gap_frames=180,
        )
    )


def _observe(
    resolver: GateDStoppageAwareGamePlayerResolverV5,
    *,
    frame_id: int,
    track_id: int,
    team_id: str = "team_a",
    active: tuple[int, ...] | None = None,
) -> str | None:
    return resolver.observe(
        frame_id=frame_id,
        track_id=track_id,
        appearance=APPEARANCE,
        team_id=team_id,
        team_confidence=0.95,
        simultaneous_track_ids=(track_id,) if active is None else active,
    )


def _seed(resolver: GateDStoppageAwareGamePlayerResolverV5) -> str:
    first = _observe(resolver, frame_id=1, track_id=1)
    second = _observe(resolver, frame_id=2, track_id=1)
    assert first is not None
    assert second == first
    return first


def _freeze(resolver: GateDStoppageAwareGamePlayerResolverV5, start: int, end: int) -> None:
    for frame_id in range(start, end + 1):
        resolver.freeze_non_gameplay(frame_id)


def test_same_raw_track_survives_long_frozen_stoppage_without_track_gap_split() -> None:
    resolver = _resolver()
    identity = _seed(resolver)

    _freeze(resolver, 3, 180)
    resolver.resume_gameplay(181)
    resumed = _observe(resolver, frame_id=181, track_id=1)

    assert resumed == identity
    episode = resolver._episode_for_track(1)
    assert episode is not None
    assert episode.episode_index == 0
    assert episode.frame_end == 181
    resolution = resolver.resolution(1, 181)
    assert resolution is not None
    assert resolution["episode_open_reason"] == "initial_track_observation"


def test_frozen_frames_are_not_added_to_episode_resolution_interval() -> None:
    resolver = _resolver()
    _seed(resolver)

    _freeze(resolver, 3, 40)
    resolver.resume_gameplay(41)
    _observe(resolver, frame_id=41, track_id=1)

    # Memory survives, but stoppage frames are not fabricated as identity observations.
    assert resolver.resolve(1, 20) is None
    assert resolver.resolve(1, 41) is not None


def test_new_raw_track_can_reassociate_across_long_frozen_stoppage() -> None:
    resolver = _resolver()
    identity = _seed(resolver)

    # Raw source gap is far beyond the 180-frame association window, but every missing
    # frame is explicitly frozen NON_GAMEPLAY and therefore must not age identity memory.
    _freeze(resolver, 3, 300)
    resolver.resume_gameplay(301)
    reassociated = _observe(resolver, frame_id=301, track_id=2)

    assert reassociated == identity
    resolution = resolver.resolution(2, 301)
    assert resolution is not None
    assert resolution["association_reason"] == "matched_existing_entity"


def test_real_gameplay_gap_still_opens_track_gap_episode() -> None:
    resolver = _resolver()
    _seed(resolver)

    # No freeze_non_gameplay calls: this is a genuine observed-time gap and must retain
    # the existing Gate-D split semantics.
    _observe(resolver, frame_id=40, track_id=1)

    episode = resolver._episode_for_track(1)
    assert episode is not None
    assert episode.episode_index == 1
    resolution = resolver.resolution(1, 40)
    assert resolution is not None
    assert resolution["episode_open_reason"] == "track_gap"


def test_cross_track_different_team_remains_hard_reject_after_stoppage() -> None:
    resolver = _resolver()
    identity = _seed(resolver)

    _freeze(resolver, 3, 300)
    resolver.resume_gameplay(301)
    other = _observe(
        resolver,
        frame_id=301,
        track_id=2,
        team_id="team_b",
    )

    assert other is not None
    assert other != identity


def test_simultaneous_tracks_cannot_share_identity_after_stoppage() -> None:
    resolver = _resolver()
    identity = _seed(resolver)

    _freeze(resolver, 3, 100)
    resolver.resume_gameplay(101)
    resumed = _observe(
        resolver,
        frame_id=101,
        track_id=1,
        active=(1, 2),
    )
    other = _observe(
        resolver,
        frame_id=101,
        track_id=2,
        active=(1, 2),
    )

    assert resumed == identity
    assert other is not None
    assert other != identity


def test_stoppage_diagnostics_report_completed_range() -> None:
    resolver = _resolver()
    _seed(resolver)
    _freeze(resolver, 3, 10)

    active = resolver.stoppage_diagnostics()
    assert active["active_frozen_range"] == [3, 10]

    resolver.resume_gameplay(11)
    closed = resolver.stoppage_diagnostics()
    assert closed["active_frozen_range"] is None
    assert closed["completed_frozen_ranges"] == 1
    assert closed["completed_frozen_frames"] == 8


def main() -> None:
    test_same_raw_track_survives_long_frozen_stoppage_without_track_gap_split()
    test_frozen_frames_are_not_added_to_episode_resolution_interval()
    test_new_raw_track_can_reassociate_across_long_frozen_stoppage()
    test_real_gameplay_gap_still_opens_track_gap_episode()
    test_cross_track_different_team_remains_hard_reject_after_stoppage()
    test_simultaneous_tracks_cannot_share_identity_after_stoppage()
    test_stoppage_diagnostics_report_completed_range()
    print("game_player_resolver_gate_d_v5_stoppage_test: PASS")


if __name__ == "__main__":
    main()
