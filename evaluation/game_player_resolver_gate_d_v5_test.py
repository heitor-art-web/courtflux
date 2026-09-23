from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_resolver_gate_d_v5 import (
    GateDDeferredAssociationConfig,
    GateDGamePlayerResolverV5,
)
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


def _unit(angle_degrees: float) -> list[float]:
    angle = math.radians(angle_degrees)
    return [math.cos(angle), math.sin(angle)]


def _resolver() -> GateDGamePlayerResolverV5:
    return GateDGamePlayerResolverV5(
        GamePlayerResolverConfig(
            appearance_match_similarity=0.90,
            appearance_plausible_similarity=0.75,
            appearance_match_margin=0.06,
            appearance_split_similarity=0.60,
            split_conflict_observations=2,
            max_track_gap_frames=40,
            max_association_gap_frames=60,
        ),
        deferred_association_config=GateDDeferredAssociationConfig(
            min_episode_observations=12,
            min_consistent_observations=8,
            min_best_similarity=0.94,
            min_margin=0.025,
        ),
    )


def _seed_two_entities(
    resolver: GateDGamePlayerResolverV5,
    *,
    second_angle: float,
) -> tuple[str, str]:
    first = _unit(0.0)
    second = _unit(second_angle)
    for frame_id in range(1, 4):
        resolver.prepare_frame(
            frame_id=frame_id,
            observations=[(1, first), (2, second)],
        )
        resolver.observe(
            frame_id=frame_id,
            track_id=1,
            appearance=first,
            simultaneous_track_ids=(1, 2),
        )
        resolver.observe(
            frame_id=frame_id,
            track_id=2,
            appearance=second,
            simultaneous_track_ids=(1, 2),
        )

    first_id = resolver.resolve(1, 3)
    second_id = resolver.resolve(2, 3)
    assert first_id is not None
    assert second_id is not None
    assert first_id != second_id
    return first_id, second_id


def test_persistent_strong_ambiguous_candidate_resolves_late() -> None:
    resolver = _resolver()
    first_id, _ = _seed_two_entities(resolver, second_angle=20.0)
    target = _unit(5.0)

    result = None
    for frame_id in range(10, 30):
        resolver.prepare_frame(frame_id=frame_id, observations=[(3, target)])
        result = resolver.observe(
            frame_id=frame_id,
            track_id=3,
            appearance=target,
            simultaneous_track_ids=(3,),
        )

    assert result == first_id
    resolution = resolver.resolution(3, 29)
    assert resolution is not None
    assert resolution["association_reason"] == "matched_existing_entity_deferred"


def test_close_persistent_ambiguity_stays_unknown() -> None:
    resolver = _resolver()
    _seed_two_entities(resolver, second_angle=14.0)
    target = _unit(5.0)

    result = None
    for frame_id in range(10, 35):
        resolver.prepare_frame(frame_id=frame_id, observations=[(3, target)])
        result = resolver.observe(
            frame_id=frame_id,
            track_id=3,
            appearance=target,
            simultaneous_track_ids=(3,),
        )

    assert result is None
    resolution = resolver.resolution(3, 34)
    assert resolution is not None
    assert resolution["association_reason"] == "unknown_ambiguous_match"


def test_active_candidate_is_hard_rejected_from_deferred_ranking() -> None:
    resolver = _resolver()
    first_id, second_id = _seed_two_entities(resolver, second_angle=20.0)
    target = _unit(5.0)
    resolver.prepare_frame(frame_id=10, observations=[(3, target)])
    resolver.observe(
        frame_id=10,
        track_id=3,
        appearance=target,
        simultaneous_track_ids=(3,),
    )
    episode = resolver._episode_for_track(3)
    assert episode is not None

    previous = resolver._gate_d_active_track_ids
    resolver._gate_d_active_track_ids = {1, 3}
    try:
        ranked = resolver._rank_deferred_candidates(episode)
    finally:
        resolver._gate_d_active_track_ids = previous

    ranked_ids = {game_player_id for _, game_player_id in ranked}
    assert first_id not in ranked_ids
    assert second_id in ranked_ids


def main() -> None:
    test_persistent_strong_ambiguous_candidate_resolves_late()
    test_close_persistent_ambiguity_stays_unknown()
    test_active_candidate_is_hard_rejected_from_deferred_ranking()
    print("game_player_resolver_gate_d_v5_test: PASS")


if __name__ == "__main__":
    main()
