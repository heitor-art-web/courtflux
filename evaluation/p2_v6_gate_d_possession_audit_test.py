from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_v6_gate_d_possession_audit import build_audit


def _prediction(
    *,
    frame_id: int,
    track_id: int,
    episode_index: int,
    team: str,
    game_player_id: str | None,
    open_reason: str,
    association_reason: str,
) -> dict:
    return {
        "frame_id": frame_id,
        "track_id": track_id,
        "episode_index": episode_index,
        "team": team,
        "game_player_id": game_player_id,
        "episode_open_reason": open_reason,
        "association_reason": association_reason,
        "global_visual_shift": False,
    }


def test_audit_uses_human_blocks_and_final_episode_decision() -> None:
    states = [
        {
            "frame_id": 0,
            "timestamp_sec": 0.0,
            "team_predictions": [
                {
                    "track_id": 1,
                    "team": "team_a",
                    "recent_reversal_confirmed": False,
                    "active_on_court": True,
                }
            ],
            "game_player_predictions": [
                _prediction(
                    frame_id=0,
                    track_id=1,
                    episode_index=0,
                    team="team_a",
                    game_player_id="game_player_001",
                    open_reason="initial_track_observation",
                    association_reason="new_entity_no_compatible_history",
                )
            ],
        },
        {
            "frame_id": 1,
            "timestamp_sec": 1.0,
            "team_predictions": [
                {
                    "track_id": 1,
                    "team": "team_b",
                    "recent_reversal_confirmed": False,
                    "active_on_court": True,
                }
            ],
            "game_player_predictions": [
                _prediction(
                    frame_id=1,
                    track_id=1,
                    episode_index=1,
                    team="team_b",
                    game_player_id=None,
                    open_reason="team_conflict",
                    association_reason="unknown_weak_best_match",
                )
            ],
        },
        {
            "frame_id": 2,
            "timestamp_sec": 2.0,
            "team_predictions": [
                {
                    "track_id": 1,
                    "team": "team_b",
                    "recent_reversal_confirmed": False,
                    "active_on_court": True,
                }
            ],
            "game_player_predictions": [
                _prediction(
                    frame_id=2,
                    track_id=1,
                    episode_index=1,
                    team="team_b",
                    game_player_id="game_player_002",
                    open_reason="team_conflict",
                    association_reason="new_entity_strong_visual_separation",
                )
            ],
        },
        {
            "frame_id": 3,
            "timestamp_sec": 3.0,
            "team_predictions": [
                {
                    "track_id": 1,
                    "team": "team_a",
                    "recent_reversal_confirmed": False,
                    "active_on_court": True,
                }
            ],
            "game_player_predictions": [
                _prediction(
                    frame_id=3,
                    track_id=1,
                    episode_index=2,
                    team="team_a",
                    game_player_id="game_player_003",
                    open_reason="team_conflict",
                    association_reason="new_entity_strong_visual_separation",
                )
            ],
        },
    ]
    gt = {
        "video": {"source_window": "synthetic"},
        "scoring_blocks": [
            {
                "id": "P1",
                "start_sec": 0.0,
                "end_sec": 4.0,
                "gameplay": "gameplay",
                "team_id": "team_a",
            }
        ],
    }
    report = {
        "episodes": 3,
        "tracks": 1,
        "episode_open_reason_counts": {"team_conflict": 2},
        "invariants": {"simultaneous_same_game_player_conflicts": 0},
    }

    audit = build_audit(states=states, ground_truth=gt, report=report)
    assert audit["summary"]["episodes"] == 3
    assert audit["summary"]["temporal_splits"] == 2
    assert audit["summary"]["split_open_reasons"]["team_conflict"] == 2
    assert audit["summary"]["split_dispositions"]["new_entity"] == 2
    assert audit["team_conflict_audit"]["episodes"] == 2
    assert (
        audit["team_conflict_audit"]["reversal_confirmation"][
            "explicitly_not_confirmed"
        ]
        == 2
    )
    assert audit["team_conflict_audit"]["immediate_team_return_episodes"] == 1
    assert audit["per_block"][0]["temporal_splits"] == 2
    assert audit["report_consistency"]["episodes_match"] is True


def test_missing_reversal_metadata_is_not_treated_as_confirmation() -> None:
    states = [
        {
            "frame_id": 0,
            "timestamp_sec": 0.0,
            "team_predictions": [{"track_id": 2, "team": "team_b"}],
            "game_player_predictions": [
                _prediction(
                    frame_id=0,
                    track_id=2,
                    episode_index=0,
                    team="team_b",
                    game_player_id="game_player_001",
                    open_reason="team_conflict",
                    association_reason="new_entity_no_compatible_history",
                )
            ],
        }
    ]
    gt = {
        "scoring_blocks": [
            {
                "id": "P1",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "gameplay": "gameplay",
                "team_id": "team_b",
            }
        ]
    }
    audit = build_audit(states=states, ground_truth=gt)
    assert audit["team_conflict_audit"]["reversal_confirmation"] == {
        "metadata_missing": 1
    }


def main() -> None:
    test_audit_uses_human_blocks_and_final_episode_decision()
    test_missing_reversal_metadata_is_not_treated_as_confirmation()
    print("p2_v6_gate_d_possession_audit_test: PASS")


if __name__ == "__main__":
    main()
