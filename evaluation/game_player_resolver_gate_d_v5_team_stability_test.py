from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_gate_d_detector_v5 import PerformanceDetector
from services.vision.game_player_resolver_gate_d_v5_team_stable import (
    GateDTeamStableGamePlayerResolverV5,
)
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


APPEARANCE = [1.0, 0.0]


def _resolver() -> GateDTeamStableGamePlayerResolverV5:
    return GateDTeamStableGamePlayerResolverV5(
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
    resolver: GateDTeamStableGamePlayerResolverV5,
    *,
    frame_id: int,
    track_id: int = 1,
    team_id: str = "team_a",
    confirmed: bool | None = None,
    metadata_active: bool = True,
) -> str | None:
    if metadata_active:
        resolver.prepare_team_stability(
            {} if confirmed is None else {track_id: confirmed}
        )
    try:
        return resolver.observe(
            frame_id=frame_id,
            track_id=track_id,
            appearance=APPEARANCE,
            team_id=team_id,
            team_confidence=0.95,
            simultaneous_track_ids=(track_id,),
        )
    finally:
        if metadata_active:
            resolver.clear_team_stability()


def _seed_team_a(resolver: GateDTeamStableGamePlayerResolverV5) -> str:
    first = _observe(resolver, frame_id=1, team_id="team_a", confirmed=False)
    second = _observe(resolver, frame_id=2, team_id="team_a", confirmed=False)
    assert first is not None and second == first
    return first


def test_unconfirmed_opposite_team_does_not_split_or_overwrite_episode_team() -> None:
    resolver = _resolver()
    identity = _seed_team_a(resolver)

    for frame_id in range(3, 7):
        result = _observe(
            resolver,
            frame_id=frame_id,
            team_id="team_b",
            confirmed=False if frame_id == 3 else None,
        )
        assert result == identity

    episode = resolver._episode_for_track(1)
    assert episode is not None
    assert episode.episode_index == 0
    assert episode.team_id == "team_a"
    assert episode.team_conflict_streak == 0


def test_confirmed_reversal_latches_until_existing_split_rule_opens_episode() -> None:
    resolver = _resolver()
    original = _seed_team_a(resolver)

    first = _observe(
        resolver,
        frame_id=3,
        team_id="team_b",
        confirmed=True,
    )
    assert first == original
    episode = resolver._episode_for_track(1)
    assert episode is not None
    assert episode.episode_index == 0
    assert episode.team_conflict_streak == 1

    second = _observe(
        resolver,
        frame_id=4,
        team_id="team_b",
        confirmed=False,
    )
    assert second is not None
    assert second != original
    episode = resolver._episode_for_track(1)
    assert episode is not None
    assert episode.episode_index == 1
    assert episode.team_id == "team_b"

    resolution = resolver.resolution(1, 4)
    assert resolution is not None
    assert resolution["episode_open_reason"] == "team_conflict"


def test_track_gap_still_splits_and_keeps_raw_incoming_team() -> None:
    resolver = _resolver()
    original = _seed_team_a(resolver)

    result = _observe(
        resolver,
        frame_id=40,
        team_id="team_b",
        confirmed=False,
    )
    assert result is not None
    assert result != original
    episode = resolver._episode_for_track(1)
    assert episode is not None
    assert episode.episode_index == 1
    assert episode.team_id == "team_b"
    resolution = resolver.resolution(1, 40)
    assert resolution is not None
    assert resolution["episode_open_reason"] == "track_gap"


def test_no_team_stability_metadata_preserves_legacy_v5_split_behavior() -> None:
    resolver = _resolver()
    original = _observe(
        resolver,
        frame_id=1,
        team_id="team_a",
        metadata_active=False,
    )
    assert original is not None
    _observe(
        resolver,
        frame_id=2,
        team_id="team_b",
        metadata_active=False,
    )
    changed = _observe(
        resolver,
        frame_id=3,
        team_id="team_b",
        metadata_active=False,
    )
    assert changed is not None
    assert changed != original
    episode = resolver._episode_for_track(1)
    assert episode is not None
    assert episode.episode_index == 1


def test_cross_track_team_incompatibility_remains_hard_reject() -> None:
    resolver = _resolver()
    first = _seed_team_a(resolver)
    resolver.prepare_team_stability({2: False})
    try:
        second = resolver.observe(
            frame_id=3,
            track_id=2,
            appearance=APPEARANCE,
            team_id="team_b",
            team_confidence=0.95,
            simultaneous_track_ids=(2,),
        )
    finally:
        resolver.clear_team_stability()

    assert second is not None
    assert second != first


def test_detector_extracts_only_explicit_boolean_reversal_confirmations() -> None:
    frame_state = SimpleNamespace(
        diagnostics={
            "detector": {
                "team_predictions": [
                    {"track_id": 1, "recent_reversal_confirmed": True},
                    {"track_id": 2, "recent_reversal_confirmed": False},
                    {"track_id": 3},
                    {"track_id": 4, "recent_reversal_confirmed": 1},
                ]
            }
        }
    )
    assert PerformanceDetector._gate_d_team_change_confirmations(frame_state) == {
        1: True,
        2: False,
    }


def main() -> None:
    test_unconfirmed_opposite_team_does_not_split_or_overwrite_episode_team()
    test_confirmed_reversal_latches_until_existing_split_rule_opens_episode()
    test_track_gap_still_splits_and_keeps_raw_incoming_team()
    test_no_team_stability_metadata_preserves_legacy_v5_split_behavior()
    test_cross_track_team_incompatibility_remains_hard_reject()
    test_detector_extracts_only_explicit_boolean_reversal_confirmations()
    print("game_player_resolver_gate_d_v5_team_stability_test: PASS")


if __name__ == "__main__":
    main()
