from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_team_colour_crop_audit import (
    _box_for_track,
    _find_state,
    _load_jsonl,
    _read_frame,
    _safe_crop,
    _team_torso_bounds,
    _timestamp,
    _video_path,
)
from services.vision.team_only_analyzer import TeamOnlyAnalyzer


def _prediction_for_track(row: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for item in list(row.get("team_predictions") or []):
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if current == track_id:
            matches.append(item)
    for item in matches:
        if item.get("rescored") is True and isinstance(item.get("raw_scores"), dict):
            return item
    return matches[0] if matches else None


def _r(value: float | np.floating[Any], places: int = 6) -> float:
    return round(float(value), places)


def _lab_stats(lab_pixels: np.ndarray) -> dict[str, Any]:
    mean = np.mean(lab_pixels, axis=0)
    median = np.median(lab_pixels, axis=0)
    # Coarse 16-level Lab mode: useful as a robust description of the densest camera-space colour.
    quantized = np.clip(np.floor(lab_pixels / 16.0), 0, 15).astype(np.int16)
    unique, counts = np.unique(quantized, axis=0, return_counts=True)
    mode_index = int(np.argmax(counts)) if counts.size else 0
    dominant = (unique[mode_index].astype(np.float32) * 16.0 + 8.0) if counts.size else np.zeros(3)
    return {
        "mean_lab": [_r(value, 3) for value in mean],
        "median_lab": [_r(value, 3) for value in median],
        "dominant_lab_bin_center": [_r(value, 3) for value in dominant],
        "dominant_lab_bin_pixels": int(counts[mode_index]) if counts.size else 0,
    }


def diagnose_torso(analyzer: TeamOnlyAnalyzer, torso: np.ndarray) -> dict[str, Any]:
    if torso is None or torso.size == 0:
        raise ValueError("torso ROI is empty")

    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    lab_pixels = lab.reshape(-1, 3).astype(np.float32)
    total_pixels = int(lab_pixels.shape[0])
    team_names = [str(team) for team in analyzer.active_colour_prototypes]
    if not team_names:
        raise ValueError("analyzer has no active teams")

    team_affinities: list[np.ndarray] = []
    within_team_winner: dict[str, np.ndarray] = {}
    prototype_rows: list[dict[str, Any]] = []
    global_distances: list[np.ndarray] = []
    global_keys: list[str] = []

    for team_name in team_names:
        prototypes = list(analyzer.active_colour_prototypes.get(team_name, []) or [])
        weights = list(analyzer.prototype_weights.get(team_name, [1.0] * len(prototypes)))
        weighted: list[np.ndarray] = []
        for index, prototype in enumerate(prototypes):
            proto_lab = np.asarray(prototype["lab"], dtype=np.float32)
            distance = np.linalg.norm(lab_pixels - proto_lab.reshape(1, 3), axis=1)
            affinity = analyzer._affinity(lab_pixels, proto_lab)
            weight = float(weights[index])
            key = f"{team_name}:{prototype['label']}:{index}"
            weighted.append(affinity * weight)
            global_distances.append(distance)
            global_keys.append(key)
            prototype_rows.append(
                {
                    "key": key,
                    "team": team_name,
                    "label": str(prototype["label"]),
                    "rgb": [int(value) for value in prototype["rgb"]],
                    "lab": [_r(value, 3) for value in proto_lab],
                    "weight": _r(weight),
                    "distance_min": _r(np.min(distance), 3),
                    "distance_p10": _r(np.percentile(distance, 10), 3),
                    "distance_median": _r(np.median(distance), 3),
                    "distance_mean": _r(np.mean(distance), 3),
                    "affinity_mean": _r(np.mean(affinity)),
                    "affinity_top_mean": _r(analyzer._top_mean(affinity)),
                    "affinity_max": _r(np.max(affinity)),
                    "discriminative_vote_pixels": 0,
                    "discriminative_vote_mass": 0.0,
                }
            )

        if weighted:
            stack = np.stack(weighted, axis=1)
            team_affinities.append(np.max(stack, axis=1))
            within_team_winner[team_name] = np.argmax(stack, axis=1)
        else:
            team_affinities.append(np.zeros((total_pixels,), dtype=np.float32))
            within_team_winner[team_name] = np.zeros((total_pixels,), dtype=np.int64)

    matrix = np.stack(team_affinities, axis=1)
    best_indices = np.argmax(matrix, axis=1)
    best_values = matrix[np.arange(total_pixels), best_indices]
    if matrix.shape[1] > 1:
        second_values = np.partition(matrix, matrix.shape[1] - 2, axis=1)[:, -2]
    else:
        second_values = np.zeros_like(best_values)

    affinity_ok = best_values >= analyzer.PIXEL_MIN_AFFINITY
    margin_values = best_values - second_values
    margin_ok = margin_values >= analyzer.PIXEL_MIN_MARGIN
    discriminative = affinity_ok & margin_ok

    votes_by_team: Counter[str] = Counter()
    votes_by_prototype: Counter[str] = Counter()
    masses_by_team = {team: 0.0 for team in team_names}

    proto_row_by_key = {row["key"]: row for row in prototype_rows}
    for team_index, team_name in enumerate(team_names):
        team_mask = discriminative & (best_indices == team_index)
        count = int(np.count_nonzero(team_mask))
        votes_by_team[team_name] = count
        masses_by_team[team_name] = float(np.sum(best_values[team_mask])) if count else 0.0
        if not count:
            continue
        local_winners = within_team_winner[team_name][team_mask]
        local_values = best_values[team_mask]
        prototypes = list(analyzer.active_colour_prototypes.get(team_name, []) or [])
        for proto_index, prototype in enumerate(prototypes):
            proto_mask = local_winners == proto_index
            proto_count = int(np.count_nonzero(proto_mask))
            if not proto_count:
                continue
            key = f"{team_name}:{prototype['label']}:{proto_index}"
            votes_by_prototype[key] += proto_count
            proto_row_by_key[key]["discriminative_vote_pixels"] = proto_count
            proto_row_by_key[key]["discriminative_vote_mass"] = _r(np.sum(local_values[proto_mask]))

    total_mass = sum(masses_by_team.values())
    raw_scores = {
        team: (masses_by_team[team] / total_mass if total_mass > 1e-12 else 0.0)
        for team in team_names
    }

    nearest_counts: Counter[str] = Counter()
    if global_distances:
        distance_matrix = np.stack(global_distances, axis=1)
        nearest_indices = np.argmin(distance_matrix, axis=1)
        nearest_counts.update(global_keys[int(index)] for index in nearest_indices)

    discriminative_count = int(np.count_nonzero(discriminative))
    low_affinity_count = int(np.count_nonzero(~affinity_ok))
    low_margin_count = int(np.count_nonzero(affinity_ok & ~margin_ok))

    return {
        "total_pixels": total_pixels,
        "lab": _lab_stats(lab_pixels),
        "thresholds": {
            "pixel_min_affinity": float(analyzer.PIXEL_MIN_AFFINITY),
            "pixel_min_margin": float(analyzer.PIXEL_MIN_MARGIN),
            "lab_sigma": float(analyzer.LAB_SIGMA),
        },
        "discriminative_pixels": discriminative_count,
        "discriminative_pixel_fraction": _r(discriminative_count / max(total_pixels, 1)),
        "abstained_pixels": total_pixels - discriminative_count,
        "rejected_low_affinity_pixels": low_affinity_count,
        "rejected_low_margin_pixels": low_margin_count,
        "votes_by_team": dict(votes_by_team),
        "votes_by_prototype": dict(votes_by_prototype),
        "evidence_mass_by_team": {team: _r(value) for team, value in masses_by_team.items()},
        "raw_scores": {team: _r(value) for team, value in raw_scores.items()},
        "nearest_unweighted_prototype_all_pixels": dict(nearest_counts),
        "prototypes": prototype_rows,
    }


def _parse_case(raw: str) -> tuple[str, str, int, int, str]:
    parts = raw.split(":", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "--case must be LABEL:RUN_ID:FRAME_ID:TRACK_ID:MATCH_CONFIG"
        )
    label, run_id, frame_raw, track_raw, match_config = parts
    try:
        frame_id = int(frame_raw)
        track_id = int(track_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame and track must be integers") from exc
    if not label.strip() or not run_id.strip() or not match_config.strip():
        raise argparse.ArgumentTypeError("label, run id, and match config cannot be empty")
    return label.strip(), run_id.strip(), frame_id, track_id, match_config.strip()


def export_case(
    *,
    label: str,
    run_id: str,
    frame_id: int,
    track_id: int,
    match_config: str,
) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    summary_path = run_dir / "summary.json"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    config_path = Path(match_config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    states = _load_jsonl(states_path)
    row = _find_state(states, frame_id)
    box = _box_for_track(row, track_id)
    timestamp_sec = _timestamp(row)
    video_path, video_source = _video_path(summary_path, run_dir)
    frame = _read_frame(video_path, timestamp_sec)
    player, clipped_box = _safe_crop(frame, box)
    left, top, right, bottom = _team_torso_bounds(player)
    torso = player[top:bottom, left:right].copy()

    analyzer = TeamOnlyAnalyzer(str(config_path))
    diagnostic = diagnose_torso(analyzer, torso)
    recomputed = analyzer.score_teams(player)
    archived = _prediction_for_track(row, track_id) or {}
    archived_raw = {
        str(key): _r(float(value))
        for key, value in dict(archived.get("raw_scores") or {}).items()
    }
    recomputed_raw = {
        str(key): _r(float(value))
        for key, value in dict(recomputed.get("raw_scores") or {}).items()
    }
    score_delta = {
        team: _r(recomputed_raw.get(team, 0.0) - archived_raw.get(team, 0.0))
        for team in sorted(set(archived_raw) | set(recomputed_raw))
    }

    return {
        "label": label,
        "run_id": run_id,
        "frame_id": frame_id,
        "track_id": track_id,
        "timestamp_sec": _r(timestamp_sec),
        "match_config": str(config_path.relative_to(PROJECT_ROOT)) if config_path.is_relative_to(PROJECT_ROOT) else str(config_path),
        "video_source": video_source,
        "video_path": str(video_path),
        "player_box": clipped_box,
        "torso_roi_in_player": [left, top, right, bottom],
        "archived_prediction": {
            "rescored": archived.get("rescored"),
            "best_team": archived.get("best_team"),
            "raw_scores": archived_raw,
        },
        "recomputed_prediction": {
            "best_team": recomputed.get("best_team"),
            "evidence_accepted": recomputed.get("evidence_accepted"),
            "raw_scores": recomputed_raw,
            "margin": recomputed.get("margin"),
        },
        "recomputed_minus_archived_raw_score": score_delta,
        "pixel_diagnostic": diagnostic,
    }


def _print_case(result: dict[str, Any]) -> None:
    diag = result["pixel_diagnostic"]
    print(f"\n===== {result['label']} / frame={result['frame_id']} track={result['track_id']} =====")
    print(f"archived_raw={result['archived_prediction']['raw_scores']}")
    print(f"recomputed_raw={result['recomputed_prediction']['raw_scores']}")
    print(f"raw_score_delta={result['recomputed_minus_archived_raw_score']}")
    print(
        "pixels="
        f"{diag['total_pixels']} discriminative={diag['discriminative_pixels']} "
        f"abstained={diag['abstained_pixels']} low_affinity={diag['rejected_low_affinity_pixels']} "
        f"low_margin={diag['rejected_low_margin_pixels']}"
    )
    print(f"lab={diag['lab']}")
    print(f"votes_by_team={diag['votes_by_team']}")
    print(f"votes_by_prototype={diag['votes_by_prototype']}")
    print(f"mass_by_team={diag['evidence_mass_by_team']}")
    print(f"nearest_unweighted_all_pixels={diag['nearest_unweighted_prototype_all_pixels']}")
    print("PROTOTYPES:")
    for proto in diag["prototypes"]:
        print(
            {
                "key": proto["key"],
                "rgb": proto["rgb"],
                "lab": proto["lab"],
                "weight": proto["weight"],
                "distance_p10": proto["distance_p10"],
                "distance_median": proto["distance_median"],
                "affinity_top_mean": proto["affinity_top_mean"],
                "vote_pixels": proto["discriminative_vote_pixels"],
                "vote_mass": proto["discriminative_vote_mass"],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain Gate-C team-colour decisions at pixel/prototype level."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_parse_case,
        metavar="LABEL:RUN_ID:FRAME_ID:TRACK_ID:MATCH_CONFIG",
        help="Repeat for each diagnostic crop.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/gate_c_colour_space_audit.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    results = [
        export_case(
            label=label,
            run_id=run_id,
            frame_id=frame_id,
            track_id=track_id,
            match_config=match_config,
        )
        for label, run_id, frame_id, track_id, match_config in args.case
    ]

    payload = {
        "version": "courtflux_p2_team_colour_space_audit_v1",
        "contract": "diagnostic-only; reconstructs the exact player torso ROI and explains current continuous CIELAB discriminative scoring without changing inference or Gate-C thresholds",
        "cases": results,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("===== TEAM COLOUR SPACE AUDIT =====")
    for result in results:
        _print_case(result)
    print(f"\n[COLOUR_SPACE_AUDIT] {output}")


if __name__ == "__main__":
    main()
