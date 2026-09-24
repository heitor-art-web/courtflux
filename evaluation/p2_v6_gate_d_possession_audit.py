from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GT = PROJECT_ROOT / "evaluation" / "ground_truth" / "v6.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _prediction_rows(states: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        frame_id = int(state.get("frame_id", -1))
        timestamp = float(state.get("timestamp_sec", 0.0) or 0.0)
        for raw in list(state.get("game_player_predictions", []) or []):
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            row.setdefault("frame_id", frame_id)
            row["timestamp_sec"] = timestamp
            rows.append(row)
    return rows


def _episode_rows(states: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _prediction_rows(states):
        try:
            key = (int(row.get("track_id", -1)), int(row.get("episode_index")))
        except (TypeError, ValueError):
            continue
        if key[0] < 0:
            continue
        grouped[key].append(row)

    episodes: list[dict[str, Any]] = []
    for (track_id, episode_index), rows in grouped.items():
        rows.sort(key=lambda item: int(item.get("frame_id", -1)))
        first, last = rows[0], rows[-1]
        episodes.append(
            {
                "track_id": track_id,
                "episode_index": episode_index,
                "frame_start": int(first.get("frame_id", -1)),
                "frame_end": int(last.get("frame_id", -1)),
                "timestamp_start_sec": float(first.get("timestamp_sec", 0.0) or 0.0),
                "timestamp_end_sec": float(last.get("timestamp_sec", 0.0) or 0.0),
                "observations": len(rows),
                "game_player_id": last.get("game_player_id"),
                "team_start": first.get("team"),
                "team_final": last.get("team"),
                "association_reason": last.get("association_reason"),
                "episode_open_reason": first.get("episode_open_reason"),
                "global_visual_shift": bool(first.get("global_visual_shift", False)),
            }
        )
    episodes.sort(key=lambda row: (row["frame_start"], row["track_id"], row["episode_index"]))
    return episodes


def _scoring_blocks(gt: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = [
        dict(row)
        for row in list(gt.get("scoring_blocks", []) or [])
        if isinstance(row, dict)
    ]
    blocks.sort(key=lambda row: float(row["start_sec"]))
    if not blocks:
        raise ValueError("V6 ground truth has no scoring_blocks")
    return blocks


def _block_for_timestamp(
    timestamp_sec: float,
    blocks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for index, block in enumerate(blocks):
        start = float(block["start_sec"])
        end = float(block["end_sec"])
        final = index == len(blocks) - 1
        if start <= timestamp_sec < end or (final and start <= timestamp_sec <= end):
            return block
    return None


def _split_disposition(association_reason: Any) -> str:
    reason = str(association_reason or "")
    if reason.startswith("new_entity"):
        return "new_entity"
    if reason.startswith("matched_existing_entity"):
        return "reassociated"
    return "unknown"


def _team_prediction_at_open(
    episode: dict[str, Any],
    state_by_frame: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    state = state_by_frame.get(int(episode["frame_start"]))
    if state is None:
        return None
    for row in list(state.get("team_predictions", []) or []):
        if not isinstance(row, dict):
            continue
        try:
            track_id = int(row.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id == int(episode["track_id"]):
            return row
    return None


def _context_for_block(block: dict[str, Any] | None) -> str:
    if block is None:
        return "outside_gt"
    gameplay = str(block.get("gameplay") or "")
    team_id = block.get("team_id")
    if gameplay == "non_game":
        return "non_game"
    if gameplay == "free_throw":
        return "free_throw"
    if team_id in {"team_a", "team_b"}:
        return "single_team_gameplay"
    return "ambiguous_gameplay"


def build_audit(
    *,
    states: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocks = _scoring_blocks(ground_truth)
    episodes = _episode_rows(states)
    state_by_frame = {
        int(row.get("frame_id", -1)): row
        for row in states
        if int(row.get("frame_id", -1)) >= 0
    }

    per_block: list[dict[str, Any]] = []
    for block in blocks:
        block_id = str(block["id"])
        rows = [
            episode
            for episode in episodes
            if (
                (matched := _block_for_timestamp(
                    float(episode["timestamp_start_sec"]), blocks
                ))
                is not None
                and str(matched["id"]) == block_id
            )
        ]
        splits = [
            row
            for row in rows
            if str(row.get("episode_open_reason") or "")
            != "initial_track_observation"
        ]
        per_block.append(
            {
                "block_id": block_id,
                "start_sec": float(block["start_sec"]),
                "end_sec": float(block["end_sec"]),
                "gameplay": block.get("gameplay"),
                "team_id": block.get("team_id"),
                "episodes_opened": len(rows),
                "unique_tracks_opened": len(
                    {int(row["track_id"]) for row in rows}
                ),
                "initial_track_episodes": sum(
                    str(row.get("episode_open_reason"))
                    == "initial_track_observation"
                    for row in rows
                ),
                "temporal_splits": len(splits),
                "split_open_reasons": dict(
                    sorted(
                        Counter(
                            str(row.get("episode_open_reason") or "unknown")
                            for row in splits
                        ).items()
                    )
                ),
                "split_dispositions": dict(
                    sorted(
                        Counter(
                            _split_disposition(row.get("association_reason"))
                            for row in splits
                        ).items()
                    )
                ),
            }
        )

    splits = [
        row
        for row in episodes
        if str(row.get("episode_open_reason") or "")
        != "initial_track_observation"
    ]
    team_conflicts = [
        row for row in episodes if row.get("episode_open_reason") == "team_conflict"
    ]

    track_episodes: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        track_episodes[int(row["track_id"])].append(row)
    for rows in track_episodes.values():
        rows.sort(key=lambda item: int(item["episode_index"]))

    team_rows: list[dict[str, Any]] = []
    bounce_durations: list[float] = []
    for episode in team_conflicts:
        prediction = _team_prediction_at_open(episode, state_by_frame)
        block = _block_for_timestamp(float(episode["timestamp_start_sec"]), blocks)
        if prediction is None or "recent_reversal_confirmed" not in prediction:
            confirmation = "metadata_missing"
        elif prediction.get("recent_reversal_confirmed") is True:
            confirmation = "confirmed"
        else:
            confirmation = "explicitly_not_confirmed"

        sequence = track_episodes[int(episode["track_id"])]
        sequence_index = next(
            (
                index
                for index, row in enumerate(sequence)
                if int(row["episode_index"]) == int(episode["episode_index"])
            ),
            -1,
        )
        immediate_return = False
        if 0 < sequence_index < len(sequence) - 1:
            previous = sequence[sequence_index - 1]
            following = sequence[sequence_index + 1]
            previous_team = previous.get("team_final")
            current_team = episode.get("team_final")
            following_team = following.get("team_final")
            immediate_return = bool(
                previous_team
                and current_team
                and following_team
                and previous_team == following_team
                and previous_team != current_team
            )
        duration = max(
            0.0,
            float(episode["timestamp_end_sec"])
            - float(episode["timestamp_start_sec"]),
        )
        if immediate_return:
            bounce_durations.append(duration)

        team_rows.append(
            {
                "track_id": int(episode["track_id"]),
                "episode_index": int(episode["episode_index"]),
                "frame_start": int(episode["frame_start"]),
                "timestamp_start_sec": round(
                    float(episode["timestamp_start_sec"]), 6
                ),
                "block_id": None if block is None else block.get("id"),
                "context": _context_for_block(block),
                "final_episode_team": episode.get("team_final"),
                "association_reason": episode.get("association_reason"),
                "reversal_confirmation": confirmation,
                "recent_reversal_candidate": (
                    None
                    if prediction is None
                    else prediction.get("recent_reversal_candidate")
                ),
                "recent_reversal_streak": (
                    None
                    if prediction is None
                    else prediction.get("recent_reversal_streak")
                ),
                "active_on_court": (
                    None if prediction is None else prediction.get("active_on_court")
                ),
                "global_visual_shift": bool(episode.get("global_visual_shift", False)),
                "immediate_team_return": immediate_return,
                "episode_duration_sec": round(duration, 6),
            }
        )

    confirmation_counts = Counter(
        str(row["reversal_confirmation"]) for row in team_rows
    )
    context_counts = Counter(str(row["context"]) for row in team_rows)
    active_counts = Counter(
        "true" if row["active_on_court"] is True else "not_true"
        for row in team_rows
    )

    audit: dict[str, Any] = {
        "version": "courtflux_v6_gate_d_possession_audit_v1",
        "contract": (
            "Diagnostic Gate-D fragmentation audit aligned to the frozen V6 human "
            "possession/non-game blocks. It does not invent SAME/DIFFERENT identity labels."
        ),
        "source_window": dict(ground_truth.get("video", {}) or {}).get(
            "source_window"
        ),
        "summary": {
            "episodes": len(episodes),
            "unique_tracks": len({int(row["track_id"]) for row in episodes}),
            "initial_track_episodes": sum(
                row.get("episode_open_reason") == "initial_track_observation"
                for row in episodes
            ),
            "temporal_splits": len(splits),
            "split_open_reasons": dict(
                sorted(
                    Counter(
                        str(row.get("episode_open_reason") or "unknown")
                        for row in splits
                    ).items()
                )
            ),
            "split_dispositions": dict(
                sorted(
                    Counter(
                        _split_disposition(row.get("association_reason"))
                        for row in splits
                    ).items()
                )
            ),
            "simultaneous_identity_conflicts": (
                None
                if report is None
                else int(
                    dict(report.get("invariants", {}) or {}).get(
                        "simultaneous_same_game_player_conflicts", 0
                    )
                    or 0
                )
            ),
        },
        "per_block": per_block,
        "team_conflict_audit": {
            "episodes": len(team_rows),
            "unique_tracks": len({int(row["track_id"]) for row in team_rows}),
            "reversal_confirmation": dict(sorted(confirmation_counts.items())),
            "context": dict(sorted(context_counts.items())),
            "active_on_court": dict(sorted(active_counts.items())),
            "global_visual_shift_opens": sum(
                bool(row["global_visual_shift"]) for row in team_rows
            ),
            "immediate_team_return_episodes": sum(
                bool(row["immediate_team_return"]) for row in team_rows
            ),
            "immediate_team_return_duration_sec": {
                "min": None
                if not bounce_durations
                else round(min(bounce_durations), 6),
                "median": None
                if not bounce_durations
                else round(median(bounce_durations), 6),
                "mean": None
                if not bounce_durations
                else round(mean(bounce_durations), 6),
                "max": None
                if not bounce_durations
                else round(max(bounce_durations), 6),
            },
            "rows": team_rows,
        },
        "report_consistency": None,
        "notes": [
            (
                "new_entity after a temporal split is diagnostic fragmentation, not by "
                "itself a human-verified identity error"
            ),
            (
                "team_conflict is a resolver cause label; this audit separately checks "
                "whether Identity-v2 had actually confirmed a team reversal"
            ),
        ],
    }

    if report is not None:
        audit["report_consistency"] = {
            "report_episodes": int(report.get("episodes", 0) or 0),
            "audited_episodes": len(episodes),
            "episodes_match": int(report.get("episodes", 0) or 0) == len(episodes),
            "report_tracks": int(report.get("tracks", 0) or 0),
            "audited_tracks": len({int(row["track_id"]) for row in episodes}),
            "tracks_match": int(report.get("tracks", 0) or 0)
            == len({int(row["track_id"]) for row in episodes}),
            "report_team_conflict_opens": int(
                dict(report.get("episode_open_reason_counts", {}) or {}).get(
                    "team_conflict", 0
                )
                or 0
            ),
            "audited_team_conflict_opens": len(team_rows),
        }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V6 Gate-D fragmentation against the human possession/non-game blocks."
        )
    )
    parser.add_argument("states", type=Path)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = None if args.report is None else read_json(args.report)
    payload = build_audit(
        states=read_jsonl(args.states),
        ground_truth=read_json(args.gt),
        report=report,
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
