from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_resolver_gate_d_v3 import GateDGamePlayerResolverV3
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


def _resolver() -> GateDGamePlayerResolverV3:
    return GateDGamePlayerResolverV3(
        GamePlayerResolverConfig(
            appearance_match_similarity=0.90,
            appearance_plausible_similarity=0.75,
            appearance_match_margin=0.05,
            appearance_split_similarity=0.60,
            split_conflict_observations=2,
            max_track_gap_frames=10,
            max_association_gap_frames=20,
        )
    )


def _descriptor(columns: tuple[int | None, int | None, int | None]) -> list[float]:
    values = [0.0] * (6 * 3 * 6)
    for row in range(6):
        for col, feature in enumerate(columns):
            if feature is not None:
                values[((row * 3 + col) * 6) + feature] = 1.0
    return values


def _observe_frame(
    resolver: GateDGamePlayerResolverV3,
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


def _seed_four_tracks(resolver: GateDGamePlayerResolverV3) -> list[float]:
    stable = _descriptor((0, 0, None))
    for frame_id in range(1, 7):
        _observe_frame(
            resolver,
            frame_id,
            {1: stable, 2: stable, 3: stable, 4: stable},
        )
    return stable


def test_isolated_raw_conflict_is_not_global_shift() -> None:
    resolver = _resolver()
    stable = _seed_four_tracks(resolver)
    switched = _descriptor((0, 1, None))

    context = _observe_frame(
        resolver,
        7,
        {1: switched, 2: stable, 3: stable, 4: stable},
    )

    assert context["global_visual_shift"] is False
    assert context["coarse_conflicting_tracks"] == 0
    assert context["raw_conflicting_tracks"] == 1


def test_raw_conflict_cohort_is_detected_and_rebased() -> None:
    resolver = _resolver()
    _seed_four_tracks(resolver)
    switched = _descriptor((0, 1, None))

    context = _observe_frame(
        resolver,
        7,
        {1: switched, 2: switched, 3: switched, 4: switched},
    )
    assert context["global_visual_shift"] is True
    assert context["comparable_tracks"] == 4
    assert context["conflicting_tracks"] == 4
    assert context["coarse_conflicting_tracks"] == 0
    assert context["raw_conflicting_tracks"] == 4

    second_context = _observe_frame(
        resolver,
        8,
        {1: switched, 2: switched, 3: switched, 4: switched},
    )
    assert second_context["global_visual_shift"] is False
    for track_id in (1, 2, 3, 4):
        assert len([key for key in resolver.episodes if key[0] == track_id]) == 1


def main() -> None:
    test_isolated_raw_conflict_is_not_global_shift()
    test_raw_conflict_cohort_is_detected_and_rebased()
    print("game_player_resolver_gate_d_v3_test: PASS")


if __name__ == "__main__":
    main()
