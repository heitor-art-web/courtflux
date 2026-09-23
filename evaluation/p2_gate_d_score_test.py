from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_gate_d_score import score_clip


def _write_states(path: Path) -> None:
    rows = [
        {
            "frame_id": 10,
            "game_player_predictions": [
                {"track_id": 1, "episode_index": 0, "game_player_id": "game_player_001"},
                {"track_id": 2, "episode_index": 0, "game_player_id": "game_player_002"},
            ],
        },
        {
            "frame_id": 20,
            "game_player_predictions": [
                {"track_id": 3, "episode_index": 0, "game_player_id": "game_player_001"},
                {"track_id": 4, "episode_index": 0, "game_player_id": "game_player_002"},
            ],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_strict_mapping_and_relation_scoring() -> None:
    payload = {
        "observations": [
            {
                "id": "SAME-1",
                "clip": "v1",
                "label": "SAME",
                "endpoints": [
                    {"frame_id": 10, "track_id": 1, "episode_index": 0},
                    {"frame_id": 20, "track_id": 3, "episode_index": 0},
                ],
            },
            {
                "id": "DIFF-1",
                "clip": "v1",
                "label": "DIFFERENT",
                "endpoints": [
                    {"frame_id": 10, "track_id": 1, "episode_index": 0},
                    {"frame_id": 20, "track_id": 3, "episode_index": 0},
                ],
            },
            {
                "id": "UNMAPPED",
                "clip": "v1",
                "label": "DIFFERENT",
                "endpoints": None,
            },
        ]
    }

    with tempfile.TemporaryDirectory() as tmp:
        states = Path(tmp) / "states.jsonl"
        _write_states(states)
        result = score_clip(payload, clip="v1", states_path=states)

    assert result["same_pairs"] == 1
    assert result["same_pairs_resolved"] == 1
    assert result["same_pairs_correct"] == 1
    assert result["same_player_continuity_recall"] == 1.0
    assert result["different_pairs"] == 2
    assert result["different_pairs_resolved"] == 1
    assert result["different_false_merges"] == 1
    assert result["different_false_merge_rate"] == 0.5
    assert result["mapping_required"] == 1
    assert result["observations"][2]["status"] == "mapping_required"
    assert result["observations"][2]["correct"] is None


def main() -> None:
    test_strict_mapping_and_relation_scoring()
    print("p2_gate_d_score_test: PASS")


if __name__ == "__main__":
    main()
