from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERESTING_REASONS = {
    "ambiguous_single_digit_tens",
    "raw_not_in_roster",
    "low_raw_consensus_support",
    "low_raw_consensus_posterior",
    "low_raw_consensus_margin",
    "team_unknown_for_roster_validation",
}
HIGH_QUALITY_THRESHOLD = 0.80


def _json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _resolve_path(value: str | None, *, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return (base / path).resolve()


def _mean_quality(prediction: dict[str, Any]) -> float:
    values = [
        float(value)
        for value in list(prediction.get("crop_quality") or [])
        if isinstance(value, (int, float))
    ]
    return float(sum(values) / len(values)) if values else 0.0


def _top_text(values: Any) -> str:
    result = []
    for row in list(values or [])[:3]:
        if not isinstance(row, dict):
            continue
        result.append(
            f"{row.get('digit', '?')}:{float(row.get('probability', 0.0) or 0.0):.2f}"
        )
    return " ".join(result) or "n/a"


def _record(frame_row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    mean_quality = _mean_quality(prediction)
    accepted = bool(prediction.get("accepted"))
    reason = str(prediction.get("rejection_reason") or ("accepted" if accepted else "unknown"))
    return {
        "frame_id": int(frame_row.get("frame_id", prediction.get("frame_id", -1)) or -1),
        "timestamp_sec": float(frame_row.get("timestamp_sec", 0.0) or 0.0),
        "track_id": int(prediction.get("track_id", -1) or -1),
        "team": prediction.get("team"),
        "team_confidence": prediction.get("team_confidence"),
        "team_margin": prediction.get("team_margin"),
        "accepted": accepted,
        "rejection_reason": reason,
        "candidate_number": prediction.get("candidate_number"),
        "raw_canonical": prediction.get("raw_canonical"),
        "candidate_probability": prediction.get("candidate_probability"),
        "candidate_posterior": prediction.get("candidate_posterior"),
        "candidate_margin": prediction.get("candidate_margin"),
        "candidate_count": prediction.get("candidate_count"),
        "top_candidates": prediction.get("top_candidates"),
        "tens_top3": prediction.get("tens_top3"),
        "ones_top3": prediction.get("ones_top3"),
        "single_digit_gate": prediction.get("single_digit_gate"),
        "roster_valid": prediction.get("roster_valid"),
        "valid_roster_numbers": prediction.get("valid_roster_numbers"),
        "identity_action": prediction.get("identity_action"),
        "identity_locked": prediction.get("identity_locked"),
        "mean_crop_quality": round(mean_quality, 6),
        "high_quality_abstention": bool(not accepted and mean_quality >= HIGH_QUALITY_THRESHOLD),
        "crop_quality": prediction.get("crop_quality"),
        "per_crop_predictions": prediction.get("per_crop_predictions"),
    }


def _box_for_track(row: dict[str, Any], track_id: int) -> list[int] | None:
    for prediction in list(row.get("team_predictions") or []):
        if not isinstance(prediction, dict):
            continue
        try:
            current = int(prediction.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        box = prediction.get("box")
        if current == track_id and isinstance(box, list) and len(box) == 4:
            return [int(value) for value in box]
    return None


def _read_video_frame(cap: cv2.VideoCapture, timestamp_sec: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
    ok, frame = cap.read()
    return frame if ok and isinstance(frame, np.ndarray) and frame.size else None


def _torso_crop(frame: np.ndarray, box: list[int]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    player = frame[y1:y2, x1:x2]
    ph, pw = player.shape[:2]
    ymin, ymax = int(ph * 0.15), int(ph * 0.65)
    xmin, xmax = int(pw * 0.10), int(pw * 0.90)
    torso = player[ymin:ymax, xmin:xmax]
    return torso if torso.size else None


def _fit_image(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    if image is None or image.size == 0:
        cv2.putText(
            canvas,
            "crop unavailable",
            (18, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return canvas
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    rh, rw = resized.shape[:2]
    x = (width - rw) // 2
    y = (height - rh) // 2
    canvas[y:y + rh, x:x + rw] = resized
    return canvas


def _draw_contact_sheet(
    output_path: Path,
    record: dict[str, Any],
    rows_by_frame: dict[int, dict[str, Any]],
    cap: cv2.VideoCapture | None,
) -> bool:
    per_crop = list(record.get("per_crop_predictions") or [])[:8]
    if not per_crop:
        return False

    columns, rows = 4, 2
    tile_w, tile_h = 310, 300
    image_h = 205
    header_h = 116
    sheet = np.full((header_h + rows * tile_h, columns * tile_w, 3), 18, dtype=np.uint8)

    header_lines = [
        f"track {record['track_id']} | {record.get('timestamp_sec', 0.0):.2f}s | {record.get('team') or 'UNKNOWN'}",
        f"candidate #{record.get('candidate_number') or '?'} | decision {'ACCEPT' if record.get('accepted') else 'UNKNOWN'} | {record.get('rejection_reason')}",
        f"T {_top_text(record.get('tens_top3'))} | O {_top_text(record.get('ones_top3'))} | posterior {float(record.get('candidate_posterior') or 0):.3f} margin {float(record.get('candidate_margin') or 0):.3f}",
    ]
    for index, text in enumerate(header_lines):
        cv2.putText(
            sheet,
            text[:155],
            (14, 29 + index * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    track_id = int(record.get("track_id", -1))
    for index in range(rows * columns):
        row_index, column_index = divmod(index, columns)
        x0, y0 = column_index * tile_w, header_h + row_index * tile_h
        cv2.rectangle(
            sheet,
            (x0, y0),
            (x0 + tile_w - 1, y0 + tile_h - 1),
            (70, 70, 70),
            1,
        )
        if index >= len(per_crop):
            continue
        crop_meta = per_crop[index] if isinstance(per_crop[index], dict) else {}
        frame_id = int(crop_meta.get("frame_id", -1) or -1)
        source_row = rows_by_frame.get(frame_id, {})
        timestamp = float(source_row.get("timestamp_sec", 0.0) or 0.0)
        crop_image = None
        if cap is not None:
            frame = _read_video_frame(cap, timestamp)
            box = _box_for_track(source_row, track_id)
            if frame is not None and box is not None:
                crop_image = _torso_crop(frame, box)
        fitted = _fit_image(crop_image, tile_w - 16, image_h)
        sheet[y0 + 8:y0 + 8 + image_h, x0 + 8:x0 + tile_w - 8] = fitted

        quality = float(crop_meta.get("crop_quality", 0.0) or 0.0)
        raw = crop_meta.get("raw_canonical")
        lines = [
            f"f{frame_id} q={quality:.2f} raw={raw if raw is not None else '?'}",
            f"T {_top_text(crop_meta.get('tens_top3'))}",
            f"O {_top_text(crop_meta.get('ones_top3'))}",
        ]
        for line_index, text in enumerate(lines):
            cv2.putText(
                sheet,
                text[:46],
                (x0 + 10, y0 + image_h + 32 + line_index * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(
        cv2.imwrite(
            str(output_path),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
    )


def _should_export_contact(
    record: dict[str, Any],
    last_by_key: dict[tuple[int, str], int],
    *,
    gap_frames: int = 50,
) -> bool:
    reason = str(record.get("rejection_reason") or "")
    interesting = bool(
        record.get("accepted")
        or reason in INTERESTING_REASONS
        or record.get("high_quality_abstention")
    )
    if not interesting:
        return False
    track_id = int(record.get("track_id", -1))
    frame_id = int(record.get("frame_id", -1))
    key = (track_id, reason)
    previous = last_by_key.get(key)
    if previous is not None and frame_id - previous < gap_frames:
        return False
    last_by_key[key] = frame_id
    return True


def build_identity_debug(
    summary_path: str | Path,
    output_root: str | Path,
    *,
    max_cases: int = 80,
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
    records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()

    debug_jsonl = output_root / "identity_debug.jsonl"
    with debug_jsonl.open("w", encoding="utf-8") as handle:
        for frame_row in frame_rows:
            for prediction in list(frame_row.get("jersey_predictions") or []):
                if not isinstance(prediction, dict):
                    continue
                item = _record(frame_row, prediction)
                records.append(item)
                rejection_counts[str(item["rejection_reason"])] += 1
                candidate = item.get("candidate_number")
                if candidate is not None:
                    candidate_counts[str(candidate)] += 1
                handle.write(
                    json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                )

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

    contact_root = output_root / "contact_sheets"
    contact_root.mkdir(parents=True, exist_ok=True)
    contact_paths: list[Path] = []
    last_by_key: dict[tuple[int, str], int] = {}
    try:
        for item in records:
            if len(contact_paths) >= max(0, int(max_cases)):
                break
            if not _should_export_contact(item, last_by_key):
                continue
            index = len(contact_paths) + 1
            safe_reason = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in str(item["rejection_reason"])
            )[:42]
            path = contact_root / (
                f"{index:03d}_f{int(item['frame_id']):05d}_"
                f"t{int(item['track_id']):04d}_{safe_reason}.jpg"
            )
            if _draw_contact_sheet(path, item, rows_by_frame, cap):
                contact_paths.append(path)
    finally:
        if cap is not None:
            cap.release()

    archive_path = output_root / "identity_debug_contact_sheets.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in contact_paths:
            archive.write(path, arcname=f"contact_sheets/{path.name}")

    accepted = sum(1 for item in records if item.get("accepted"))
    high_quality_abstentions = sum(
        1 for item in records if item.get("high_quality_abstention")
    )
    result = {
        "enabled": True,
        "source_summary": str(summary_path),
        "frames_file": str(frames_path),
        "video_path": str(video_path) if video_path is not None else None,
        "video_available": bool(video_path is not None and video_path.exists()),
        "probability_detail": "runtime_top3_from_frames_jsonl",
        "attempts": len(records),
        "accepted": accepted,
        "abstained": len(records) - accepted,
        "high_quality_threshold": HIGH_QUALITY_THRESHOLD,
        "high_quality_abstentions": high_quality_abstentions,
        "rejection_reasons": dict(rejection_counts.most_common()),
        "candidate_counts": dict(candidate_counts.most_common()),
        "contact_sheets": len(contact_paths),
        "contact_sheet_limit": int(max_cases),
        "identity_debug_jsonl": str(debug_jsonl),
        "contact_sheets_zip": str(archive_path),
    }
    (output_root / "identity_debug_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Jersey/Identity debug traces and contact sheets."
    )
    parser.add_argument("--summary", required=True, help="Path to a completed run summary.json")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for identity debug artifacts",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=80,
        help="Maximum contact sheets to export",
    )
    args = parser.parse_args()
    result = build_identity_debug(args.summary, args.output, max_cases=args.max_cases)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
