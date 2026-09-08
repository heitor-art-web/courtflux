from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from evaluation.ball_error_review import (
    align_production_truth,
    candidate_center,
    candidate_matches_truth,
    load_jsonl,
)


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def pct(n: int, d: int) -> float:
    return round(100.0 * n / max(d, 1), 2)


def point_distance(a: tuple[float, float], b: list[float]) -> float:
    return math.hypot(a[0] - float(b[0]), a[1] - float(b[1]))


def best_direct_center(
    production: dict[str, Any], truth: dict[str, Any], radius_px: float
) -> tuple[float, float] | None:
    candidates = [
        candidate
        for candidate in ((production.get("ball_association") or {}).get("candidates") or [])
        if isinstance(candidate, dict)
    ]
    matches: list[tuple[float, tuple[float, float]]] = []
    gt_point = truth.get("point")
    for candidate in candidates:
        ok, distance = candidate_matches_truth(candidate, truth, radius_px, 0.25)
        center = candidate_center(candidate)
        if ok and center is not None:
            matches.append((float(distance or 0.0), center))
    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def evaluate_pair(
    frames_path: Path,
    annotations_path: Path,
    *,
    radius_px: float,
    gap_limits: list[int],
    timestamp_tolerance_sec: float,
) -> dict[str, Any]:
    production_rows = load_jsonl(frames_path)
    truth_rows = load_jsonl(annotations_path)
    aligned, unmatched = align_production_truth(
        production_rows, truth_rows, tolerance_sec=timestamp_tolerance_sec
    )
    rows: list[dict[str, Any]] = []
    for production, truth, delta in aligned:
        if not bool(truth.get("visible")):
            continue
        gt_point = truth.get("point")
        if not isinstance(gt_point, list) or len(gt_point) != 2:
            continue
        rows.append(
            {
                "frame_id": int(production.get("frame_id", -1)),
                "timestamp_sec": float(production.get("timestamp_sec", 0.0)),
                "timestamp_delta_sec": float(delta),
                "gt": [float(gt_point[0]), float(gt_point[1])],
                "direct": best_direct_center(production, truth, radius_px),
            }
        )

    direct_count = sum(row["direct"] is not None for row in rows)
    coverage: dict[str, Any] = {
        "direct": {
            "correct_frames": direct_count,
            "visible_frames": len(rows),
            "coverage_pct": pct(direct_count, len(rows)),
        }
    }

    for max_gap in sorted(set(max(0, int(value)) for value in gap_limits)):
        covered = [row["direct"] is not None for row in rows]
        bridged = 0
        bridge_errors: list[float] = []
        direct_indices = [idx for idx, row in enumerate(rows) if row["direct"] is not None]
        for left_idx, right_idx in zip(direct_indices, direct_indices[1:]):
            missing = right_idx - left_idx - 1
            if missing <= 0 or missing > max_gap:
                continue
            # Do not bridge across frame discontinuities caused by skipped/non-gameplay frames.
            frame_span = rows[right_idx]["frame_id"] - rows[left_idx]["frame_id"]
            if frame_span != missing + 1:
                continue
            left = rows[left_idx]["direct"]
            right = rows[right_idx]["direct"]
            if left is None or right is None:
                continue
            all_correct = True
            local_errors: list[float] = []
            for offset in range(1, missing + 1):
                alpha = offset / float(missing + 1)
                interp = (
                    float(left[0]) + alpha * (float(right[0]) - float(left[0])),
                    float(left[1]) + alpha * (float(right[1]) - float(left[1])),
                )
                idx = left_idx + offset
                error = point_distance(interp, rows[idx]["gt"])
                local_errors.append(error)
                if error > radius_px:
                    all_correct = False
            # Conservative bridge: count the gap only if every interpolated point is correct.
            if all_correct:
                for offset in range(1, missing + 1):
                    idx = left_idx + offset
                    if not covered[idx]:
                        covered[idx] = True
                        bridged += 1
                bridge_errors.extend(local_errors)

        covered_count = sum(covered)
        coverage[f"direct_plus_linear_gap_{max_gap}"] = {
            "correct_frames": covered_count,
            "visible_frames": len(rows),
            "coverage_pct": pct(covered_count, len(rows)),
            "bridged_frames": bridged,
            "bridge_error_mean_px": (
                None if not bridge_errors else round(sum(bridge_errors) / len(bridge_errors), 4)
            ),
            "method": "oracle_validated_linear_bridge_upper_bound",
        }

    return {
        "frames_path": str(frames_path),
        "annotations_path": str(annotations_path),
        "aligned_visible_frames": len(rows),
        "unmatched_gameplay_frames": unmatched,
        "gt_radius_px": radius_px,
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure direct and short-gap ball track coverage against frozen CourtFlux GT."
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--gt-radius-px", type=float, default=40.0)
    parser.add_argument("--gap", action="append", type=int, dest="gaps")
    parser.add_argument("--timestamp-tolerance-sec", type=float, default=0.001)
    parser.add_argument("--output", default="evaluation/ball_track_coverage")
    args = parser.parse_args()

    gaps = args.gaps or [1, 2, 3, 5, 8]
    run_root = Path(args.run_root).resolve()
    gt_root = Path(args.gt_root).resolve()
    results: list[dict[str, Any]] = []
    for run_id, gt_id in args.pair:
        result = evaluate_pair(
            run_root / run_id / "frames.jsonl",
            gt_root / gt_id / "annotations.jsonl",
            radius_px=float(args.gt_radius_px),
            gap_limits=gaps,
            timestamp_tolerance_sec=float(args.timestamp_tolerance_sec),
        )
        result["run_id"] = run_id
        result["gt_run_id"] = gt_id
        results.append(result)

    aggregate: dict[str, Any] = {
        "scope": "CourtFlux ball track coverage / detector-target evidence",
        "metric_note": (
            "Linear bridges are GT-validated upper bounds for short interpolation gaps, not production tracker results."
        ),
        "runs": results,
        "aggregate": {},
    }
    keys = ["direct"] + [f"direct_plus_linear_gap_{gap}" for gap in sorted(set(gaps))]
    for key in keys:
        total_correct = sum(int(run["coverage"][key]["correct_frames"]) for run in results)
        total_visible = sum(int(run["coverage"][key]["visible_frames"]) for run in results)
        aggregate["aggregate"][key] = {
            "correct_frames": total_correct,
            "visible_frames": total_visible,
            "coverage_pct": pct(total_correct, total_visible),
        }

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"[TRACK COVERAGE] {path}")


if __name__ == "__main__":
    main()
