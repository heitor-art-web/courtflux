from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Endpoint:
    frame_id: int
    track_id: int
    episode_index: int | None = None


@dataclass(frozen=True, slots=True)
class Resolution:
    game_player_id: str | None
    episode_index: int | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_states(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _prediction_rows(state_row: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = state_row.get("game_player_predictions", [])
    if not isinstance(predictions, list):
        return []
    return [row for row in predictions if isinstance(row, dict)]


def _index_states(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    index: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for state in rows:
        frame_id = int(state.get("frame_id", -1) or -1)
        for prediction in _prediction_rows(state):
            track_id = int(prediction.get("track_id", -1) or -1)
            if frame_id < 0 or track_id < 0:
                continue
            index.setdefault((frame_id, track_id), []).append(prediction)
    return index


def _parse_endpoint(raw: Any) -> Endpoint:
    if not isinstance(raw, dict):
        raise ValueError("endpoint must be an object")
    return Endpoint(
        frame_id=int(raw["frame_id"]),
        track_id=int(raw["track_id"]),
        episode_index=(None if raw.get("episode_index") is None else int(raw["episode_index"])),
    )


def _resolve_endpoint(
    index: dict[tuple[int, int], list[dict[str, Any]]], endpoint: Endpoint
) -> Resolution | None:
    candidates = index.get((endpoint.frame_id, endpoint.track_id), [])
    if endpoint.episode_index is not None:
        candidates = [
            row
            for row in candidates
            if row.get("episode_index") is not None
            and int(row["episode_index"]) == endpoint.episode_index
        ]
    if len(candidates) != 1:
        return None
    row = candidates[0]
    episode_index = row.get("episode_index")
    return Resolution(
        game_player_id=(None if row.get("game_player_id") is None else str(row["game_player_id"])),
        episode_index=(None if episode_index is None else int(episode_index)),
    )


def _score_observation(
    observation: dict[str, Any], index: dict[tuple[int, int], list[dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": observation.get("id"),
        "clip": observation.get("clip"),
        "label": observation.get("label"),
    }
    endpoints_raw = observation.get("endpoints")
    if not isinstance(endpoints_raw, list) or len(endpoints_raw) != 2:
        result.update({"status": "mapping_required", "resolved": False, "correct": None})
        return result

    endpoints = [_parse_endpoint(raw) for raw in endpoints_raw]
    resolutions = [_resolve_endpoint(index, endpoint) for endpoint in endpoints]
    if any(resolution is None for resolution in resolutions):
        result.update({"status": "endpoint_not_found", "resolved": False, "correct": None})
        return result

    left, right = resolutions
    assert left is not None and right is not None
    result["endpoint_resolutions"] = [
        {"game_player_id": left.game_player_id, "episode_index": left.episode_index},
        {"game_player_id": right.game_player_id, "episode_index": right.episode_index},
    ]
    if left.game_player_id is None or right.game_player_id is None:
        result.update({"status": "resolver_unknown", "resolved": False, "correct": None})
        return result

    same_prediction = left.game_player_id == right.game_player_id
    label = str(observation.get("label", "")).upper()
    if label not in {"SAME", "DIFFERENT"}:
        raise ValueError(f"Unsupported label for {observation.get('id')}: {label}")
    correct = same_prediction if label == "SAME" else not same_prediction
    result.update(
        {
            "status": "scored",
            "resolved": True,
            "predicted_relation": "SAME" if same_prediction else "DIFFERENT",
            "correct": bool(correct),
        }
    )
    return result


def score_clip(
    observations_payload: dict[str, Any], *, clip: str, states_path: Path
) -> dict[str, Any]:
    rows = _load_states(states_path)
    index = _index_states(rows)
    observations = [
        item
        for item in list(observations_payload.get("observations", []) or [])
        if isinstance(item, dict) and str(item.get("clip")) == clip
    ]
    scored = [_score_observation(observation, index) for observation in observations]

    same = [row for row in scored if row.get("label") == "SAME"]
    different = [row for row in scored if row.get("label") == "DIFFERENT"]
    same_resolved = [row for row in same if row.get("resolved")]
    different_resolved = [row for row in different if row.get("resolved")]
    same_correct = [row for row in same_resolved if row.get("correct") is True]
    false_merges = [
        row
        for row in different_resolved
        if row.get("predicted_relation") == "SAME"
    ]

    return {
        "version": "courtflux_gate_d_score_v1",
        "clip": clip,
        "states_file": str(states_path),
        "observations_total": len(scored),
        "mapping_required": sum(row.get("status") == "mapping_required" for row in scored),
        "same_pairs": len(same),
        "same_pairs_resolved": len(same_resolved),
        "same_pairs_correct": len(same_correct),
        "same_player_continuity_recall": (
            None if not same else round(len(same_correct) / len(same), 6)
        ),
        "different_pairs": len(different),
        "different_pairs_resolved": len(different_resolved),
        "different_false_merges": len(false_merges),
        "different_false_merge_rate": (
            None if not different else round(len(false_merges) / len(different), 6)
        ),
        "observations": scored,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score frozen Gate-D SAME/DIFFERENT observations.")
    parser.add_argument("--clip", required=True, choices=("v1", "v3", "v4", "v5"))
    parser.add_argument("--states", required=True, type=Path)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("evaluation/ground_truth/gate_d_human_observations_v1.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _load_json(args.observations)
    result = score_clip(payload, clip=args.clip, states_path=args.states)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
