from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_v6_gate_d_anonymous_global_preflight import (
    MicroTracklet,
    Observation,
    _assign_global,
    _simultaneous_conflicts,
)


def _obs(frame: int, raw: int, center_x: float, *, team: str = "team_a") -> Observation:
    width = 50.0
    height = 100.0
    return Observation(
        frame_id=frame,
        timestamp_sec=frame / 60.0,
        raw_track_id=raw,
        box=(center_x - width / 2.0, 100.0, center_x + width / 2.0, 200.0),
        team=team,
        team_confidence=0.99,
        active_on_court=True,
    )


def _pre_tracklet(index: int, raw: int, first_x: float, end_x: float) -> MicroTracklet:
    rows = [_obs(4, raw, first_x)] + [_obs(frame, raw, end_x) for frame in range(5, 10)]
    return MicroTracklet(
        index=index,
        raw_track_id=raw,
        observations=tuple(rows),
        start_reason="raw_track_start",
        best_team="team_a",
        team_posterior=0.99,
    )


def _post_tracklet(index: int, raw: int, start_x: float, end_x: float) -> MicroTracklet:
    rows = [_obs(frame, raw, start_x) for frame in range(10, 15)] + [_obs(15, raw, end_x)]
    return MicroTracklet(
        index=index,
        raw_track_id=raw,
        observations=tuple(rows),
        start_reason="crossing_exit",
        best_team="team_a",
        team_posterior=0.99,
    )


def test_global_search_can_swap_identity_across_raw_track_crossing() -> None:
    # Two people converge at x=100. After the encounter, raw IDs are intentionally
    # swapped: raw=1 follows the left-moving human and raw=2 follows the right-moving
    # human. A correct anonymous association therefore crosses raw IDs.
    a_pre = _pre_tracklet(0, raw=1, first_x=50.0, end_x=100.0)
    b_pre = _pre_tracklet(1, raw=2, first_x=150.0, end_x=100.0)
    b_post_on_raw1 = _post_tracklet(2, raw=1, start_x=90.0, end_x=40.0)
    a_post_on_raw2 = _post_tracklet(3, raw=2, start_x=110.0, end_x=160.0)
    tracklets = [a_pre, b_pre, b_post_on_raw1, a_post_on_raw2]

    assignments, _diag = _assign_global(
        tracklets,
        team_names=["team_a", "team_b"],
        slots_per_team=2,
        beam_width=64,
    )

    assert assignments[0] != assignments[1]
    assert assignments[2] == assignments[1]
    assert assignments[3] == assignments[0]
    assert _simultaneous_conflicts(tracklets, assignments) == []


def test_strong_team_evidence_keeps_tracklet_in_team_slot_pool() -> None:
    tracklet = MicroTracklet(
        index=0,
        raw_track_id=5,
        observations=tuple(_obs(frame, 5, 50.0, team="team_b") for frame in range(1, 7)),
        start_reason="raw_track_start",
        best_team="team_b",
        team_posterior=0.99,
    )
    assignments, diag = _assign_global(
        [tracklet],
        team_names=["team_a", "team_b"],
        slots_per_team=2,
        beam_width=16,
    )
    slot = assignments[0]
    assert diag["slot_teams"][slot] == "team_b"


def main() -> None:
    test_global_search_can_swap_identity_across_raw_track_crossing()
    test_strong_team_evidence_keeps_tracklet_in_team_slot_pool()
    print("p2_v6_gate_d_anonymous_global_preflight_test: PASS")


if __name__ == "__main__":
    main()
