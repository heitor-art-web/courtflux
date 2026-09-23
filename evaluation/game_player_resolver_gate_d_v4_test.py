from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_resolver_gate_d_v4 import GateDGamePlayerResolverV4
from services.vision.game_player_resolver_v1 import GamePlayerResolverConfig


def _resolver() -> GateDGamePlayerResolverV4:
    return GateDGamePlayerResolverV4(
        GamePlayerResolverConfig(
            appearance_match_similarity=0.90,
            appearance_plausible_similarity=0.75,
            appearance_match_margin=0.05,
            appearance_split_similarity=0.60,
            split_conflict_observations=2,
            max_track_gap_frames=40,
            max_association_gap_frames=60,
        )
    )


def _descriptor(columns: tuple[int | None, int | None, int | None]) -> list[float]:
    values = [0.0] * (6 * 3 * 6)
    for row in range(6):
        for col, feature in enumerate(columns):
            if feature is not None:
                values[((row * 3 + col) * 6) + feature] = 1.0
    return values


def _seed(resolver: GateDGamePlayerResolverV4) -> tuple[list[float], list[float]]:
    stable = _descriptor((0, 0, None))
    changed = _descriptor((0, 1, None))
    for frame_id in range(1, 7):
        resolver.prepare_frame(
            frame_id=frame_id,
            observations=[(track_id, stable) for track_id in (1, 2, 3, 4)],
        )
        for track_id in (1, 2, 3, 4):
            resolver.observe(
                frame_id=frame_id,
                track_id=track_id,
                appearance=stable,
                simultaneous_track_ids=(1, 2, 3, 4),
            )
    return stable, changed


def test_isolated_conflict_does_not_start_temporal_cohort() -> None:
    resolver = _resolver()
    stable, changed = _seed(resolver)
    context = resolver.prepare_frame(
        frame_id=10,
        observations=[(1, changed), (2, stable), (3, stable), (4, stable)],
    )
    assert context["global_visual_shift"] is False
    assert context["temporal_cohort_tracks"] == 1


def test_staggered_conflicts_trigger_temporal_cohort() -> None:
    resolver = _resolver()
    stable, changed = _seed(resolver)

    first = resolver.prepare_frame(
        frame_id=10,
        observations=[(1, changed), (2, stable), (3, stable), (4, stable)],
    )
    second = resolver.prepare_frame(
        frame_id=20,
        observations=[(1, stable), (2, changed), (3, stable), (4, stable)],
    )
    third = resolver.prepare_frame(
        frame_id=30,
        observations=[(1, stable), (2, stable), (3, changed), (4, stable)],
    )

    assert first["global_visual_shift"] is False
    assert second["global_visual_shift"] is False
    assert third["global_visual_shift"] is True
    assert third["temporal_cohort_triggered"] is True
    assert third["temporal_cohort_tracks"] == 3
    assert third["conflicting_tracks"] >= 3


def test_temporal_cohort_cooldown_suppresses_following_wave() -> None:
    resolver = _resolver()
    stable, changed = _seed(resolver)
    resolver.prepare_frame(
        frame_id=10,
        observations=[(1, changed), (2, stable), (3, stable), (4, stable)],
    )
    resolver.prepare_frame(
        frame_id=20,
        observations=[(1, stable), (2, changed), (3, stable), (4, stable)],
    )
    resolver.prepare_frame(
        frame_id=30,
        observations=[(1, stable), (2, stable), (3, changed), (4, stable)],
    )
    cooldown = resolver.prepare_frame(
        frame_id=45,
        observations=[(1, stable), (2, stable), (3, stable), (4, stable)],
    )
    expired = resolver.prepare_frame(
        frame_id=61,
        observations=[(1, stable), (2, stable), (3, stable), (4, stable)],
    )

    assert cooldown["global_visual_shift"] is True
    assert cooldown["temporal_cohort_active"] is True
    assert expired["global_visual_shift"] is False


def main() -> None:
    test_isolated_conflict_does_not_start_temporal_cohort()
    test_staggered_conflicts_trigger_temporal_cohort()
    test_temporal_cohort_cooldown_suppresses_following_wave()
    print("game_player_resolver_gate_d_v4_test: PASS")


if __name__ == "__main__":
    main()
