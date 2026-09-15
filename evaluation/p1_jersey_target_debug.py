from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from evaluation.p1_jersey_probe import (
    PROJECT_ROOT,
    _draw_contact_sheet,
    _json_lines,
    _record,
    _resolve_path,
)

DEFAULT_TARGETS = ("23", "21", "13")


def _digit_probability(top3: Any, digit: int) -> float | None:
    for row in list(top3 or [])[:3]:
        if not isinstance(row, dict):
            continue
        try:
            current = int(row.get("digit"))
        except (TypeError, ValueError):
            continue
        if current == int(digit):
            return float(row.get("probability", 0.0) or 0.0)
    return None


def _target_digits(number: str) -> tuple[int, int]:
    canonical = str(int(str(number).strip()))
    if len(canonical) == 1:
        return 0, int(canonical)
    if len(canonical) == 2:
        return int(canonical[0]), int(canonical[1])
    raise ValueError(f"Unsupported jersey target {number!r}; expected 0-99")


def _target_evidence(record: dict[str, Any], target: str) -> dict[str, Any]:
    tens_digit, ones_digit = _target_digits(target)
    tens_probability = _digit_probability(record.get("tens_top3"), tens_digit)
    ones_probability = _digit_probability(record.get("ones_top3"), ones_digit)

    crop_count = 0
    tens_crop_support = 0
    ones_crop_support = 0
    both_crop_support = 0
    per_crop: list[dict[str, Any]] = []
    for row in list(record.get("per_crop_predictions") or []):
        if not isinstance(row, dict):
            continue
        crop_count += 1
        tens_value = _digit_probability(row.get("tens_top3"), tens_digit)
        ones_value = _digit_probability(row.get("ones_top3"), ones_digit)
        tens_present = tens_value is not None
        ones_present = ones_value is not None
        tens_crop_support += int(tens_present)
        ones_crop_support += int(ones_present)
        both_crop_support += int(tens_present and ones_present)
        per_crop.append(
            {
                "frame_id": row.get("frame_id"),
                "crop_quality": row.get("crop_quality"),
                "tens_probability_if_top3": tens_value,
                "ones_probability_if_top3": ones_value,
                "both_digits_in_top3": bool(tens_present and ones_present),
            }
        )

    if tens_probability is None and ones_probability is None:
        aggregate_status = "both_target_digits_missing_from_top3"
    elif tens_probability is None:
        aggregate_status = "target_tens_missing_from_top3"
    elif ones_probability is None:
        aggregate_status = "target_ones_missing_from_top3"
    else:
        aggregate_status = "both_target_digits_in_top3"

    pair_proxy = (
        float(tens_probability) * float(ones_probability)
        if tens_probability is not None and ones_probability is not None
        else 0.0
    )
    valid = {str(value) for value in list(record.get("valid_roster_numbers") or [])}
    return {
        "target_number": target,
        "target_tens_digit": tens_digit,
        "target_ones_digit": ones_digit,
        "target_is_valid_for_resolved_team": target in valid,
        "aggregate_status": aggregate_status,
        "aggregate_tens_probability_if_top3": tens_probability,
        "aggregate_ones_probability_if_top3": ones_probability,
        "aggregate_pair_proxy": round(pair_proxy, 8),
        "crop_count": crop_count,
        "tens_crop_top3_support": tens_crop_support,
        "ones_crop_top3_support": ones_crop_support,
        "both_digits_crop_top3_support": both_crop_support,
        "tens_crop_top3_ratio": round(tens_crop_support / crop_count, 6) if crop_count else 0.0,
        "ones_crop_top3_ratio": round(ones_crop_support / crop_count, 6) if crop_count else 0.0,
        "both_digits_crop_top3_ratio": round(both_crop_support / crop_count, 6) if crop_count else 0.0,
        "raw_candidate_matches_target": str(record.get("candidate_number") or "") == target,
        "accepted_target": bool(record.get("accepted")) and str(record.get("candidate_number") or "") == target,
        "per_crop_target_evidence": per_crop,
    }


def _rank_key(item: dict[str, Any]) -> tuple[float, ...]:
    evidence = item["target_evidence"]
    both = evidence.get("aggregate_status") == "both_target_digits_in_top3"
    return (
        float(bool(both)),
        float(evidence.get("aggregate_pair_proxy", 0.0) or 0.0),
        float(evidence.get("both_digits_crop_top3_ratio", 0.0) or 0.0),
        float(evidence.get("tens_crop_top3_ratio", 0.0) or 0.0)
        + float(evidence.get("ones_crop_top3_ratio", 0.0) or 0.0),
        float(item.get("mean_crop_quality", 0.0) or 0.0),
    )


def _summarize_target(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if bool(row["target_evidence"].get("target_is_valid_for_resolved_team"))
    ]
    status_counts = Counter(
        str(row["target_evidence"].get("aggregate_status") or "unknown")
        for row in eligible
    )
    total_crops = sum(int(row["target_evidence"].get("crop_count", 0) or 0) for row in eligible)
    tens_crops = sum(
        int(row["target_evidence"].get("tens_crop_top3_support", 0) or 0)
        for row in eligible
    )
    ones_crops = sum(
        int(row["target_evidence"].get("ones_crop_top3_support", 0) or 0)
        for row in eligible
    )
    both_crops = sum(
        int(row["target_evidence"].get("both_digits_crop_top3_support", 0) or 0)
        for row in eligible
    )
    ranked = sorted(eligible, key=_rank_key, reverse=True)
    return {
        "target_number": target,
        "diagnostic_only": True,
        "roster_used_for_decoding": False,
        "attempts_where_target_is_valid_for_team": len(eligible),
        "raw_candidate_hits": sum(
            1 for row in eligible if row["target_evidence"].get("raw_candidate_matches_target")
        ),
        "accepted_hits": sum(
            1 for row in eligible if row["target_evidence"].get("accepted_target")
        ),
        "aggregate_status_counts": dict(status_counts.most_common()),
        "crop_observations": total_crops,
        "target_tens_in_crop_top3": tens_crops,
        "target_ones_in_crop_top3": ones_crops,
        "both_target_digits_in_same_crop_top3": both_crops,
        "target_tens_crop_top3_rate": round(tens_crops / total_crops, 6) if total_crops else 0.0,
        "target_ones_crop_top3_rate": round(ones_crops / total_crops, 6) if total_crops else 0.0,
        "both_digits_crop_top3_rate": round(both_crops / total_crops, 6) if total_crops else 0.0,
        "best_attempts": [
            {
                "frame_id": row.get("frame_id"),
                "timestamp_sec": row.get("timestamp_sec"),
                "track_id": row.get("track_id"),
                "team": row.get("team"),
                "candidate_number": row.get("candidate_number"),
                "accepted": row.get("accepted"),
                "rejection_reason": row.get("rejection_reason"),
                "mean_crop_quality": row.get("mean_crop_quality"),
                **{
                    key: value
                    for key, value in row["target_evidence"].items()
                    if key != "per_crop_target_evidence"
                },
            }
            for row in ranked[:12]
        ],
    }


def build_target_debug(
    summary_path: str | Path,
    output_root: str | Path,
    *,
    targets: list[str] | tuple[str, ...] = DEFAULT_TARGETS,
    contact_sheets_per_target: int = 8,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    frames_path = _resolve_path(summary.get("frames_file"), base=summary_path.parent)
    if frames_path is None or not frames_path.exists():
        raise FileNotFoundError(
            f"frames.jsonl not found from summary: {summary.get('frames_file')!r}"
        )

    frame_rows = _json_lines(frames_path)
    rows_by_frame = {int(row.get("frame_id", -1)): row for row in frame_rows}
    base_records: list[dict[str, Any]] = []
    for frame_row in frame_rows:
        for prediction in list(frame_row.get("jersey_predictions") or []):
            if isinstance(prediction, dict):
                base_records.append(_record(frame_row, prediction))

    normalized_targets = []
    for target in targets:
        canonical = str(int(str(target).strip()))
        if canonical not in normalized_targets:
            _target_digits(canonical)
            normalized_targets.append(canonical)

    per_target_rows: dict[str, list[dict[str, Any]]] = {}
    for target in normalized_targets:
        enriched = []
        for record in base_records:
            enriched.append(
                {
                    **record,
                    "target_evidence": _target_evidence(record, target),
                }
            )
        per_target_rows[target] = enriched

    detailed_path = output_root / "target_digit_evidence.jsonl"
    with detailed_path.open("w", encoding="utf-8") as handle:
        for target in normalized_targets:
            for row in per_target_rows[target]:
                if not row["target_evidence"].get("target_is_valid_for_resolved_team"):
                    continue
                payload = {
                    key: value
                    for key, value in row.items()
                    if key != "per_crop_predictions"
                }
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    video_path = _resolve_path(
        (
            (summary.get("extra") or {}).get("video_path")
            if isinstance(summary.get("extra"), dict)
            else None
        ),
        base=summary_path.parent,
    )
    cap: cv2.VideoCapture | None = None
    if video_path is not None and video_path.exists():
        candidate_cap = cv2.VideoCapture(str(video_path))
        if candidate_cap.isOpened():
            cap = candidate_cap
        else:
            candidate_cap.release()

    contact_root = output_root / "target_contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)
    contact_paths: list[Path] = []
    try:
        for target in normalized_targets:
            eligible = [
                row
                for row in per_target_rows[target]
                if row["target_evidence"].get("target_is_valid_for_resolved_team")
            ]
            ranked = sorted(eligible, key=_rank_key, reverse=True)
            for index, row in enumerate(ranked[: max(0, int(contact_sheets_per_target))], start=1):
                path = contact_root / (
                    f"target_{target}_{index:02d}_f{int(row['frame_id']):05d}_"
                    f"t{int(row['track_id']):04d}.jpg"
                )
                if _draw_contact_sheet(path, row, rows_by_frame, cap):
                    contact_paths.append(path)
    finally:
        if cap is not None:
            cap.release()

    archive_path = output_root / "target_contact_sheets.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in contact_paths:
            archive.write(path, arcname=f"target_contact_sheets/{path.name}")

    target_summaries = {
        target: _summarize_target(per_target_rows[target], target)
        for target in normalized_targets
    }
    result = {
        "enabled": True,
        "diagnostic_only": True,
        "roster_policy": "validator_only",
        "roster_used_for_decoding": False,
        "note": (
            "Target numbers are inspected only to measure whether their component digits "
            "appear in the recognizer top-3 evidence. This never changes runtime decoding."
        ),
        "source_summary": str(summary_path),
        "frames_file": str(frames_path),
        "video_path": str(video_path) if video_path is not None else None,
        "targets": normalized_targets,
        "attempts": len(base_records),
        "targets_summary": target_summaries,
        "target_digit_evidence_jsonl": str(detailed_path),
        "contact_sheets": len(contact_paths),
        "target_contact_sheets_zip": str(archive_path),
    }
    summary_output = output_root / "target_debug_summary.json"
    summary_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect whether known Jersey targets such as 23/21/13 are present in the "
            "digit-head evidence without using the roster to decode them."
        )
    )
    parser.add_argument("--summary", required=True, help="Path to a completed run summary.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated jersey numbers to inspect (default: 23,21,13)",
    )
    parser.add_argument(
        "--contact-sheets-per-target",
        type=int,
        default=8,
        help="Maximum ranked contact sheets per target",
    )
    args = parser.parse_args()
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    result = build_target_debug(
        args.summary,
        args.output,
        targets=targets,
        contact_sheets_per_target=args.contact_sheets_per_target,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
