from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _prediction_rows(state_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for state in state_rows:
        try:
            frame_id = int(state.get("frame_id", 0))
        except (TypeError, ValueError):
            continue
        timestamp = float(state.get("timestamp_sec", 0.0) or 0.0)
        for raw in list(state.get("game_player_predictions", []) or []):
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            row.setdefault("frame_id", frame_id)
            row["timestamp_sec"] = timestamp
            flattened.append(row)
    return flattened


def _mode(values: Iterable[Any]) -> Any:
    clean = [value for value in values if value is not None]
    return Counter(clean).most_common(1)[0][0] if clean else None


def _max_optional(values: Iterable[Any]) -> float | None:
    clean: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            clean.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(clean) if clean else None


def build_gate_d_report(state_rows: Iterable[dict[str, Any]], *, clip: str | None = None) -> dict[str, Any]:
    predictions = _prediction_rows(state_rows)
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        try:
            track_id = int(row.get("track_id", -1))
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if track_id < 0 or frame_id < 0:
            continue
        by_track[track_id].append(row)
        by_frame[frame_id].append(row)

    segments: list[dict[str, Any]] = []
    for track_id, rows in sorted(by_track.items()):
        rows.sort(key=lambda item: int(item.get("frame_id", 0)))
        current: list[dict[str, Any]] = []
        current_key: tuple[Any, Any] | None = None
        for row in rows:
            key = (row.get("episode_index"), row.get("game_player_id"))
            if current and key != current_key:
                segments.append(_segment(track_id, current))
                current = []
            current_key = key
            current.append(row)
        if current:
            segments.append(_segment(track_id, current))

    simultaneous_conflicts: list[dict[str, Any]] = []
    for frame_id, rows in sorted(by_frame.items()):
        by_identity: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            game_player_id = row.get("game_player_id")
            if game_player_id is None:
                continue
            by_identity[str(game_player_id)].add(int(row.get("track_id", -1)))
        for game_player_id, track_ids in by_identity.items():
            if len(track_ids) > 1:
                simultaneous_conflicts.append(
                    {
                        "frame_id": frame_id,
                        "game_player_id": game_player_id,
                        "track_ids": sorted(track_ids),
                    }
                )

    entities: dict[str, dict[str, Any]] = {}
    for segment in segments:
        game_player_id = segment.get("game_player_id")
        if game_player_id is None:
            continue
        item = entities.setdefault(
            str(game_player_id),
            {"game_player_id": str(game_player_id), "track_ids": set(), "segments": 0},
        )
        item["track_ids"].add(int(segment["track_id"]))
        item["segments"] += 1
    entity_rows = [
        {
            "game_player_id": value["game_player_id"],
            "track_ids": sorted(value["track_ids"]),
            "segments": int(value["segments"]),
        }
        for value in entities.values()
    ]
    entity_rows.sort(key=lambda item: item["game_player_id"])

    resolved = sum(1 for row in predictions if row.get("game_player_id") is not None)
    total = len(predictions)
    association_reason_counts = Counter(
        str(row.get("association_reason"))
        for row in predictions
        if row.get("association_reason") is not None
    )
    episode_open_reason_counts = Counter(
        str(segment.get("episode_open_reason"))
        for segment in segments
        if segment.get("episode_open_reason") is not None
    )
    return {
        "version": "courtflux_gate_d_report_v2",
        "clip": clip,
        "prediction_observations": total,
        "resolved_observations": resolved,
        "unknown_observations": total - resolved,
        "resolved_observation_rate": round((resolved / total) if total else 0.0, 6),
        "tracks": len(by_track),
        "game_player_entities": len(entity_rows),
        "association_reason_counts": dict(sorted(association_reason_counts.items())),
        "episode_open_reason_counts": dict(sorted(episode_open_reason_counts.items())),
        "segments": segments,
        "entities": entity_rows,
        "invariants": {
            "simultaneous_same_game_player_conflicts": len(simultaneous_conflicts),
            "simultaneous_conflicts": simultaneous_conflicts,
        },
        "note": (
            "This report exposes resolver output for the existing human Gate-D observations; "
            "it does not invent or replace the frozen SAME/DIFFERENT labels."
        ),
    }


def _segment(track_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = rows[0], rows[-1]
    best_score = _max_optional(row.get("association_best_score") for row in rows)
    second_score = _max_optional(row.get("association_second_score") for row in rows)
    open_similarity = first.get("episode_open_appearance_similarity")
    try:
        open_similarity = None if open_similarity is None else round(float(open_similarity), 6)
    except (TypeError, ValueError):
        open_similarity = None
    return {
        "track_id": track_id,
        "episode_index": first.get("episode_index"),
        "game_player_id": first.get("game_player_id"),
        "frame_start": int(first.get("frame_id", 0)),
        "frame_end": int(last.get("frame_id", 0)),
        "timestamp_start_sec": round(float(first.get("timestamp_sec", 0.0) or 0.0), 6),
        "timestamp_end_sec": round(float(last.get("timestamp_sec", 0.0) or 0.0), 6),
        "observations": len(rows),
        "team": _mode(row.get("team") for row in rows),
        "jersey_number": _mode(row.get("jersey_number") for row in rows),
        "appearance_available_rate": round(
            sum(bool(row.get("appearance_available", False)) for row in rows) / len(rows),
            6,
        ),
        "association_reason": _mode(row.get("association_reason") for row in rows),
        "association_best_score_max": None if best_score is None else round(best_score, 6),
        "association_second_score_max": None if second_score is None else round(second_score, 6),
        "association_candidate_count_max": int(
            max(int(row.get("association_candidate_count", 0) or 0) for row in rows)
        ),
        "episode_open_reason": first.get("episode_open_reason"),
        "episode_open_appearance_similarity": open_similarity,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Gate-D game-player diagnostics from p2_states.jsonl.")
    parser.add_argument("states", type=Path)
    parser.add_argument("--clip", choices=("v1", "v2", "v3", "v4", "v5"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_gate_d_report(read_jsonl(args.states), clip=args.clip)
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
