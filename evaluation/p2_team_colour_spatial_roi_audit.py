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


def _score_lab_pixels(analyzer: TeamOnlyAnalyzer, lab_pixels: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(lab_pixels, dtype=np.float32).reshape(-1, 3)
    team_names = [str(team) for team in analyzer.active_colour_prototypes]
    if pixels.size == 0:
        return {
            "pixels": 0,
            "discriminative_pixels": 0,
            "discriminative_fraction": 0.0,
            "best_team_raw": None,
            "best_score": 0.0,
            "margin": 0.0,
            "would_accept": False,
            "raw_scores": {team: 0.0 for team in team_names},
            "votes_by_team": {team: 0 for team in team_names},
            "mass_by_team": {team: 0.0 for team in team_names},
            "pixel_lightness": _stats(np.array([], dtype=np.float32)),
            "pixel_chroma": _stats(np.array([], dtype=np.float32)),
        }

    raw_scores, _, evidence_fraction, discriminative_count = analyzer._team_scores(pixels)

    team_affinities: list[np.ndarray] = []
    for team_name in team_names:
        prototypes = list(analyzer.active_colour_prototypes.get(team_name, []) or [])
        weights = list(analyzer.prototype_weights.get(team_name, [1.0] * len(prototypes)))
        weighted: list[np.ndarray] = []
        for index, prototype in enumerate(prototypes):
            affinity = analyzer._affinity(pixels, np.asarray(prototype["lab"], dtype=np.float32))
            weighted.append(affinity * float(weights[index]))
        if weighted:
            team_affinities.append(np.maximum.reduce(weighted))
        else:
            team_affinities.append(np.zeros((pixels.shape[0],), dtype=np.float32))

    matrix = np.stack(team_affinities, axis=1)
    best_indices = np.argmax(matrix, axis=1)
    best_values = matrix[np.arange(matrix.shape[0]), best_indices]
    if matrix.shape[1] > 1:
        second_values = np.partition(matrix, matrix.shape[1] - 2, axis=1)[:, -2]
    else:
        second_values = np.zeros_like(best_values)

    discriminative = (
        (best_values >= analyzer.PIXEL_MIN_AFFINITY)
        & ((best_values - second_values) >= analyzer.PIXEL_MIN_MARGIN)
    )

    votes: Counter[str] = Counter()
    masses = {team: 0.0 for team in team_names}
    for team_index, team_name in enumerate(team_names):
        mask = discriminative & (best_indices == team_index)
        votes[team_name] = int(np.count_nonzero(mask))
        if np.any(mask):
            masses[team_name] = float(np.sum(best_values[mask]))

    total_mass = float(sum(masses.values()))
    diagnostic_scores = {
        team: (masses[team] / total_mass if total_mass > 1e-12 else 0.0)
        for team in team_names
    }
    max_delta = max(
        (abs(float(raw_scores.get(team, 0.0)) - diagnostic_scores[team]) for team in team_names),
        default=0.0,
    )
    if max_delta > 1e-5:
        raise RuntimeError(
            "Spatial audit no longer mirrors TeamOnlyAnalyzer._team_scores; "
            f"max raw-score delta={max_delta:.6f}"
        )

    ordered = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)
    best_team, best_score = ordered[0] if ordered else (None, 0.0)
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = max(0.0, float(best_score) - float(second_score))
    would_accept = bool(
        best_team is not None
        and evidence_fraction >= analyzer.MIN_DISCRIMINATIVE_PIXEL_FRACTION
        and float(best_score) >= analyzer.MIN_TEAM_SCORE
        and margin >= analyzer.MIN_TEAM_MARGIN
    )

    ab = pixels[:, 1:3] - 128.0
    chroma = np.linalg.norm(ab, axis=1)
    return {
        "pixels": int(pixels.shape[0]),
        "discriminative_pixels": int(discriminative_count),
        "discriminative_fraction": _r(evidence_fraction),
        "best_team_raw": best_team,
        "best_score": _r(float(best_score)),
        "margin": _r(margin),
        "would_accept": would_accept,
        "raw_scores": {team: _r(float(raw_scores.get(team, 0.0))) for team in team_names},
        "votes_by_team": {team: int(votes.get(team, 0)) for team in team_names},
        "mass_by_team": {team: _r(masses[team]) for team in team_names},
        "pixel_lightness": _stats(pixels[:, 0]),
        "pixel_chroma": _stats(chroma),
    }


def _score_mask(analyzer: TeamOnlyAnalyzer, torso: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if torso is None or torso.size == 0:
        raise ValueError("torso ROI is empty")
    if mask.shape != torso.shape[:2]:
        raise ValueError("mask shape must match torso height/width")
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB).astype(np.float32)
    return _score_lab_pixels(analyzer, lab[mask])


def _grid_masks(height: int, width: int) -> list[tuple[str, np.ndarray, list[int]]]:
    if height < 3 or width < 3:
        raise ValueError("torso ROI must be at least 3x3 for spatial audit")
    row_names = ("top", "middle", "bottom")
    col_names = ("left", "center", "right")
    y_edges = np.linspace(0, height, 4, dtype=int)
    x_edges = np.linspace(0, width, 4, dtype=int)
    result: list[tuple[str, np.ndarray, list[int]]] = []
    for row_index, row_name in enumerate(row_names):
        for col_index, col_name in enumerate(col_names):
            y1, y2 = int(y_edges[row_index]), int(y_edges[row_index + 1])
            x1, x2 = int(x_edges[col_index]), int(x_edges[col_index + 1])
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            result.append((f"{row_name}_{col_name}", mask, [x1, y1, x2, y2]))
    return result


def _third_masks(height: int, width: int, *, axis: str) -> list[tuple[str, np.ndarray]]:
    if axis == "row":
        names = ("top", "middle", "bottom")
        edges = np.linspace(0, height, 4, dtype=int)
        result = []
        for index, name in enumerate(names):
            mask = np.zeros((height, width), dtype=bool)
            mask[int(edges[index]):int(edges[index + 1]), :] = True
            result.append((name, mask))
        return result
    if axis == "column":
        names = ("left", "center", "right")
        edges = np.linspace(0, width, 4, dtype=int)
        result = []
        for index, name in enumerate(names):
            mask = np.zeros((height, width), dtype=bool)
            mask[:, int(edges[index]):int(edges[index + 1])] = True
            result.append((name, mask))
        return result
    raise ValueError("axis must be row or column")


def analyze_torso_spatial(analyzer: TeamOnlyAnalyzer, torso: np.ndarray) -> dict[str, Any]:
    if torso is None or torso.size == 0:
        raise ValueError("torso ROI is empty")
    height, width = torso.shape[:2]
    full_mask = np.ones((height, width), dtype=bool)

    inner_x1 = int(round(width * 0.25))
    inner_x2 = int(round(width * 0.75))
    inner_y1 = int(round(height * 0.20))
    inner_y2 = int(round(height * 0.80))
    inner_x1 = max(0, min(width - 1, inner_x1))
    inner_x2 = max(inner_x1 + 1, min(width, inner_x2))
    inner_y1 = max(0, min(height - 1, inner_y1))
    inner_y2 = max(inner_y1 + 1, min(height, inner_y2))

    inner_mask = np.zeros((height, width), dtype=bool)
    inner_mask[inner_y1:inner_y2, inner_x1:inner_x2] = True
    outer_mask = ~inner_mask

    grid = []
    for name, mask, bounds in _grid_masks(height, width):
        grid.append({"name": name, "bounds": bounds, "score": _score_mask(analyzer, torso, mask)})

    rows = {
        name: _score_mask(analyzer, torso, mask)
        for name, mask in _third_masks(height, width, axis="row")
    }
    columns = {
        name: _score_mask(analyzer, torso, mask)
        for name, mask in _third_masks(height, width, axis="column")
    }

    return {
        "torso_size": [width, height],
        "full": _score_mask(analyzer, torso, full_mask),
        "inner_core": {
            "bounds": [inner_x1, inner_y1, inner_x2, inner_y2],
            "score": _score_mask(analyzer, torso, inner_mask),
        },
        "outer_ring": _score_mask(analyzer, torso, outer_mask),
        "rows": rows,
        "columns": columns,
        "grid_3x3": grid,
    }


def _parse_case(raw: str) -> tuple[str, str, int, int, str, str]:
    parts = raw.split(":", 5)
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "--case must be LABEL:RUN_ID:FRAME_ID:TRACK_ID:MATCH_CONFIG:EXPECTED_TEAM"
        )
    label, run_id, frame_raw, track_raw, match_config, expected_team = parts
    try:
        frame_id = int(frame_raw)
        track_id = int(track_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame and track must be integers") from exc
    if not label.strip() or not run_id.strip() or not match_config.strip() or not expected_team.strip():
        raise argparse.ArgumentTypeError("label, run id, config, and expected team cannot be empty")
    return (
        label.strip(),
        run_id.strip(),
        frame_id,
        track_id,
        match_config.strip(),
        expected_team.strip(),
    )


def _annotate_torso(torso: np.ndarray, spatial: dict[str, Any]) -> np.ndarray:
    annotated = torso.copy()
    height, width = annotated.shape[:2]
    x_edges = np.linspace(0, width, 4, dtype=int)
    y_edges = np.linspace(0, height, 4, dtype=int)
    for x in x_edges[1:-1]:
        cv2.line(annotated, (int(x), 0), (int(x), height - 1), (0, 255, 255), 1)
    for y in y_edges[1:-1]:
        cv2.line(annotated, (0, int(y)), (width - 1, int(y)), (0, 255, 255), 1)

    x1, y1, x2, y2 = spatial["inner_core"]["bounds"]
    cv2.rectangle(annotated, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)

    for cell in spatial["grid_3x3"]:
        bx1, by1, bx2, by2 = cell["bounds"]
        score = cell["score"]
        best = score["best_team_raw"] or "?"
        confidence = score["best_score"]
        cv2.putText(
            annotated,
            f"{best}:{confidence:.2f}",
            (bx1 + 2, min(by2 - 3, by1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def export_case(
    *,
    label: str,
    run_id: str,
    frame_id: int,
    track_id: int,
    match_config: str,
    expected_team: str,
    image_root: Path,
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
    spatial = analyze_torso_spatial(analyzer, torso)
    current = analyzer.score_teams(player)
    archived = _prediction_for_track(row, track_id) or {}

    current_raw = {
        str(key): _r(float(value))
        for key, value in dict(current.get("raw_scores") or {}).items()
    }
    full_raw = dict(spatial["full"]["raw_scores"])
    teams = sorted(set(current_raw) | set(full_raw))
    full_delta = {
        team: _r(full_raw.get(team, 0.0) - current_raw.get(team, 0.0))
        for team in teams
    }
    if max((abs(value) for value in full_delta.values()), default=0.0) > 1e-5:
        raise RuntimeError(
            f"full torso spatial score does not match current scorer for {label}: {full_delta}"
        )

    image_root.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
    annotated_path = image_root / f"{safe_label}_spatial.jpg"
    annotated = _annotate_torso(torso, spatial)
    if not cv2.imwrite(str(annotated_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Could not write {annotated_path}")

    archived_raw = {
        str(key): _r(float(value))
        for key, value in dict(archived.get("raw_scores") or {}).items()
    }

    return {
        "label": label,
        "run_id": run_id,
        "frame_id": frame_id,
        "track_id": track_id,
        "expected_team": expected_team,
        "timestamp_sec": _r(timestamp_sec),
        "match_config": str(config_path.relative_to(PROJECT_ROOT)) if config_path.is_relative_to(PROJECT_ROOT) else str(config_path),
        "video_source": video_source,
        "video_path": str(video_path),
        "player_box": clipped_box,
        "torso_roi_in_player": [left, top, right, bottom],
        "annotated_torso_file": str(annotated_path),
        "archived_raw_scores": archived_raw,
        "current_scorer": {
            "best_team": current.get("best_team"),
            "raw_scores": current_raw,
            "margin": current.get("margin"),
            "evidence_accepted": current.get("evidence_accepted"),
            "colour_space": current.get("colour_space"),
        },
        "full_spatial_minus_current_raw": full_delta,
        "spatial": spatial,
    }


def _compact(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "best": score["best_team_raw"],
        "raw": score["raw_scores"],
        "margin": score["margin"],
        "disc_fraction": score["discriminative_fraction"],
        "votes": score["votes_by_team"],
        "mass": score["mass_by_team"],
        "chroma_median": score["pixel_chroma"]["median"],
    }


def _print_case(result: dict[str, Any]) -> None:
    spatial = result["spatial"]
    print(
        f"\n===== {result['label']} / frame={result['frame_id']} "
        f"track={result['track_id']} truth={result['expected_team']} ====="
    )
    print(f"archived_raw={result['archived_raw_scores']}")
    print(f"current={result['current_scorer']}")
    print(f"full={_compact(spatial['full'])}")
    print(f"inner_core={_compact(spatial['inner_core']['score'])}")
    print(f"outer_ring={_compact(spatial['outer_ring'])}")
    print("ROWS:")
    for name in ("top", "middle", "bottom"):
        print({name: _compact(spatial["rows"][name])})
    print("COLUMNS:")
    for name in ("left", "center", "right"):
        print({name: _compact(spatial["columns"][name])})
    print("GRID_3X3:")
    for cell in spatial["grid_3x3"]:
        print({cell["name"]: _compact(cell["score"])})
    print(f"annotated_torso={result['annotated_torso_file']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose where Gate-C team-colour evidence lives inside the exact production torso ROI. "
            "Diagnostic only: does not change inference, GT, or thresholds."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_parse_case,
        metavar="LABEL:RUN_ID:FRAME_ID:TRACK_ID:MATCH_CONFIG:EXPECTED_TEAM",
    )
    parser.add_argument(
        "--output",
        default="evaluation/gate_c_spatial_roi_audit.json",
    )
    parser.add_argument(
        "--image-dir",
        default="evaluation/crop_audit/gate_c_spatial_roi",
    )
    args = parser.parse_args()

    image_root = Path(args.image_dir)
    if not image_root.is_absolute():
        image_root = (PROJECT_ROOT / image_root).resolve()

    results = [
        export_case(
            label=label,
            run_id=run_id,
            frame_id=frame_id,
            track_id=track_id,
            match_config=match_config,
            expected_team=expected_team,
            image_root=image_root,
        )
        for label, run_id, frame_id, track_id, match_config, expected_team in args.case
    ]

    payload = {
        "version": "courtflux_p2_team_colour_spatial_roi_audit_v1",
        "contract": (
            "diagnostic-only; exact production torso ROI, current affinity-mass scorer, "
            "inner-vs-outer and 3x3 spatial evidence; no inference/GT/Gate-C threshold changes"
        ),
        "cases": results,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = (PROJECT_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("===== TEAM COLOUR SPATIAL ROI AUDIT =====")
    for result in results:
        _print_case(result)
    print(f"\n[SPATIAL_ROI_AUDIT] {output}")


if __name__ == "__main__":
    main()
