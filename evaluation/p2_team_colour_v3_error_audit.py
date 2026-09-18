from __future__ import annotations

import argparse
import json
import math
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


def _r(value: float | np.floating[Any], places: int = 6) -> float:
    return round(float(value), places)


def _stats(values: np.ndarray, *, places: int = 3) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return {"p10": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0}
    return {
        "p10": _r(np.percentile(flat, 10), places),
        "median": _r(np.median(flat), places),
        "mean": _r(np.mean(flat), places),
        "p90": _r(np.percentile(flat, 90), places),
    }


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


def _prototype_components(
    analyzer: TeamOnlyAnalyzer,
    lab_pixels: np.ndarray,
    prototype_lab: np.ndarray,
) -> dict[str, Any]:
    pixels = lab_pixels.astype(np.float32)
    prototype = prototype_lab.astype(np.float32).reshape(3)

    pixel_l = pixels[:, 0]
    pixel_ab = pixels[:, 1:3] - 128.0
    pixel_chroma = np.linalg.norm(pixel_ab, axis=1)
    prototype_ab = prototype[1:3] - 128.0
    prototype_chroma = float(np.linalg.norm(prototype_ab))
    lightness_delta = np.abs(pixel_l - float(prototype[0]))

    if prototype_chroma < analyzer.CHROMATIC_PROTOTYPE_MIN_CHROMA:
        return {
            "model": "neutral",
            "prototype_chroma": _r(prototype_chroma, 3),
            "pixel_chroma": _stats(pixel_chroma),
            "lightness_delta": _stats(lightness_delta),
        }

    denominator = np.maximum(pixel_chroma * prototype_chroma, 1e-6)
    cosine = np.sum(pixel_ab * prototype_ab.reshape(1, 2), axis=1) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    hue_delta_rad = np.arccos(cosine)
    hue_delta_deg = np.degrees(hue_delta_rad)
    chroma_delta = np.abs(pixel_chroma - prototype_chroma)
    reliability = np.clip(
        pixel_chroma / max(analyzer.CHROMATIC_PIXEL_RELIABLE_CHROMA, 1e-6),
        0.0,
        1.0,
    )
    return {
        "model": "chromatic",
        "prototype_chroma": _r(prototype_chroma, 3),
        "pixel_chroma": _stats(pixel_chroma),
        "hue_delta_deg": _stats(hue_delta_deg),
        "chroma_delta": _stats(chroma_delta),
        "lightness_delta": _stats(lightness_delta),
        "chroma_reliability": _stats(reliability),
    }


def diagnose_v3(analyzer: TeamOnlyAnalyzer, torso: np.ndarray) -> dict[str, Any]:
    if torso is None or torso.size == 0:
        raise ValueError("torso ROI is empty")

    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    lab_pixels = lab.reshape(-1, 3).astype(np.float32)
    total_pixels = int(lab_pixels.shape[0])
    team_names = [str(team) for team in analyzer.active_colour_prototypes]
    if not team_names:
        raise ValueError("analyzer has no active teams")

    pixel_ab = lab_pixels[:, 1:3] - 128.0
    pixel_chroma = np.linalg.norm(pixel_ab, axis=1)

    team_affinities: list[np.ndarray] = []
    team_proto_winners: dict[str, np.ndarray] = {}
    prototype_rows: list[dict[str, Any]] = []
    prototype_weighted_affinities: list[np.ndarray] = []
    prototype_keys: list[str] = []

    for team_name in team_names:
        prototypes = list(analyzer.active_colour_prototypes.get(team_name, []) or [])
        weights = list(analyzer.prototype_weights.get(team_name, [1.0] * len(prototypes)))
        weighted_for_team: list[np.ndarray] = []
        for index, prototype in enumerate(prototypes):
            proto_lab = np.asarray(prototype["lab"], dtype=np.float32)
            affinity = analyzer._affinity(lab_pixels, proto_lab)
            weight = float(weights[index])
            weighted = affinity * weight
            key = f"{team_name}:{prototype['label']}:{index}"
            weighted_for_team.append(weighted)
            prototype_weighted_affinities.append(weighted)
            prototype_keys.append(key)
            prototype_rows.append(
                {
                    "key": key,
                    "team": team_name,
                    "label": str(prototype["label"]),
                    "rgb": [int(value) for value in prototype["rgb"]],
                    "lab": [_r(value, 3) for value in proto_lab],
                    "weight": _r(weight),
                    "affinity": _stats(affinity, places=6),
                    "weighted_affinity": _stats(weighted, places=6),
                    "components": _prototype_components(analyzer, lab_pixels, proto_lab),
                    "all_pixel_weighted_winner_count": 0,
                    "discriminative_vote_pixels": 0,
                    "discriminative_vote_mass": 0.0,
                }
            )

        if weighted_for_team:
            stack = np.stack(weighted_for_team, axis=1)
            team_affinities.append(np.max(stack, axis=1))
            team_proto_winners[team_name] = np.argmax(stack, axis=1)
        else:
            team_affinities.append(np.zeros((total_pixels,), dtype=np.float32))
            team_proto_winners[team_name] = np.zeros((total_pixels,), dtype=np.int64)

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

    all_proto_winners: Counter[str] = Counter()
    if prototype_weighted_affinities:
        proto_matrix = np.stack(prototype_weighted_affinities, axis=1)
        proto_best = np.argmax(proto_matrix, axis=1)
        all_proto_winners.update(prototype_keys[int(index)] for index in proto_best)

    row_by_key = {row["key"]: row for row in prototype_rows}
    for key, count in all_proto_winners.items():
        row_by_key[key]["all_pixel_weighted_winner_count"] = int(count)

    votes_by_team: Counter[str] = Counter()
    votes_by_prototype: Counter[str] = Counter()
    masses_by_team = {team: 0.0 for team in team_names}

    for team_index, team_name in enumerate(team_names):
        team_mask = discriminative & (best_indices == team_index)
        team_count = int(np.count_nonzero(team_mask))
        votes_by_team[team_name] = team_count
        masses_by_team[team_name] = float(np.sum(best_values[team_mask])) if team_count else 0.0
        if not team_count:
            continue
        local_winners = team_proto_winners[team_name][team_mask]
        local_values = best_values[team_mask]
        prototypes = list(analyzer.active_colour_prototypes.get(team_name, []) or [])
        for proto_index, prototype in enumerate(prototypes):
            proto_mask = local_winners == proto_index
            count = int(np.count_nonzero(proto_mask))
            if not count:
                continue
            key = f"{team_name}:{prototype['label']}:{proto_index}"
            votes_by_prototype[key] += count
            row_by_key[key]["discriminative_vote_pixels"] = count
            row_by_key[key]["discriminative_vote_mass"] = _r(np.sum(local_values[proto_mask]))

    total_mass = float(sum(masses_by_team.values()))
    raw_scores = {
        team: (masses_by_team[team] / total_mass if total_mass > 1e-12 else 0.0)
        for team in team_names
    }

    discriminative_count = int(np.count_nonzero(discriminative))
    return {
        "colour_space": "CIELAB_continuous_chroma_luminance_v3",
        "total_pixels": total_pixels,
        "pixel_lightness": _stats(lab_pixels[:, 0]),
        "pixel_chroma": _stats(pixel_chroma),
        "low_chroma_pixels": int(
            np.count_nonzero(pixel_chroma < analyzer.CHROMATIC_PIXEL_RELIABLE_CHROMA)
        ),
        "thresholds": {
            "pixel_min_affinity": float(analyzer.PIXEL_MIN_AFFINITY),
            "pixel_min_margin": float(analyzer.PIXEL_MIN_MARGIN),
            "chromatic_prototype_min_chroma": float(analyzer.CHROMATIC_PROTOTYPE_MIN_CHROMA),
            "chromatic_hue_sigma_rad": float(analyzer.CHROMATIC_HUE_SIGMA_RAD),
            "chromatic_chroma_sigma": float(analyzer.CHROMATIC_CHROMA_SIGMA),
            "chromatic_lightness_sigma": float(analyzer.CHROMATIC_LIGHTNESS_SIGMA),
            "chromatic_pixel_reliable_chroma": float(analyzer.CHROMATIC_PIXEL_RELIABLE_CHROMA),
            "neutral_chroma_sigma": float(analyzer.NEUTRAL_CHROMA_SIGMA),
            "neutral_lightness_sigma": float(analyzer.NEUTRAL_LIGHTNESS_SIGMA),
        },
        "discriminative_pixels": discriminative_count,
        "discriminative_pixel_fraction": _r(discriminative_count / max(total_pixels, 1)),
        "abstained_pixels": total_pixels - discriminative_count,
        "rejected_low_affinity_pixels": int(np.count_nonzero(~affinity_ok)),
        "rejected_low_margin_pixels": int(np.count_nonzero(affinity_ok & ~margin_ok)),
        "team_winner_all_pixels": {
            team_names[index]: int(np.count_nonzero(best_indices == index))
            for index in range(len(team_names))
        },
        "weighted_prototype_winner_all_pixels": dict(all_proto_winners),
        "votes_by_team": dict(votes_by_team),
        "votes_by_prototype": dict(votes_by_prototype),
        "evidence_mass_by_team": {team: _r(value) for team, value in masses_by_team.items()},
        "raw_scores": {team: _r(value) for team, value in raw_scores.items()},
        "winner_margin": _stats(margin_values, places=6),
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
    diagnostic = diagnose_v3(analyzer, torso)
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
        "archived_raw_scores": archived_raw,
        "current_scorer": {
            "best_team": recomputed.get("best_team"),
            "raw_scores": recomputed_raw,
            "margin": recomputed.get("margin"),
            "evidence_accepted": recomputed.get("evidence_accepted"),
            "colour_space": recomputed.get("colour_space"),
        },
        "pixel_diagnostic": diagnostic,
    }


def _print_case(result: dict[str, Any]) -> None:
    diag = result["pixel_diagnostic"]
    print(f"\n===== {result['label']} / frame={result['frame_id']} track={result['track_id']} =====")
    print(f"archived_raw={result['archived_raw_scores']}")
    print(f"current={result['current_scorer']}")
    print(
        f"pixels={diag['total_pixels']} discriminative={diag['discriminative_pixels']} "
        f"abstained={diag['abstained_pixels']} low_affinity={diag['rejected_low_affinity_pixels']} "
        f"low_margin={diag['rejected_low_margin_pixels']} low_chroma={diag['low_chroma_pixels']}"
    )
    print(f"pixel_lightness={diag['pixel_lightness']}")
    print(f"pixel_chroma={diag['pixel_chroma']}")
    print(f"team_winner_all_pixels={diag['team_winner_all_pixels']}")
    print(f"prototype_winner_all_pixels={diag['weighted_prototype_winner_all_pixels']}")
    print(f"votes_by_team={diag['votes_by_team']}")
    print(f"votes_by_prototype={diag['votes_by_prototype']}")
    print(f"mass_by_team={diag['evidence_mass_by_team']}")
    print("PROTOTYPES:")
    for proto in diag["prototypes"]:
        printable = {
            "key": proto["key"],
            "model": proto["components"]["model"],
            "rgb": proto["rgb"],
            "lab": proto["lab"],
            "weight": proto["weight"],
            "all_pixel_winners": proto["all_pixel_weighted_winner_count"],
            "vote_pixels": proto["discriminative_vote_pixels"],
            "vote_mass": proto["discriminative_vote_mass"],
            "affinity": proto["affinity"],
            "components": proto["components"],
        }
        print(printable)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain remaining Gate-C v3 colour errors using scorer-native hue/chroma/lightness evidence."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_parse_case,
        metavar="LABEL:RUN_ID:FRAME_ID:TRACK_ID:MATCH_CONFIG",
    )
    parser.add_argument(
        "--output",
        default="evaluation/gate_c_colour_v3_error_audit.json",
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
        "version": "courtflux_p2_team_colour_v3_error_audit_v1",
        "contract": (
            "diagnostic-only; uses the current scorer's exact hue/chroma/lightness affinity "
            "on archived run boxes without changing inference, GT, or Gate-C thresholds"
        ),
        "cases": results,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("===== TEAM COLOUR V3 ERROR AUDIT =====")
    for result in results:
        _print_case(result)
    print(f"\n[COLOUR_V3_ERROR_AUDIT] {output}")


if __name__ == "__main__":
    main()
