from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.team_only_analyzer import TeamOnlyAnalyzer

CANONICAL_MODEL = ROOT / "weights" / "model_v1.0.0.pt"
PLAYER_VISIBILITY = {"clear", "partial", "unreadable"}
NON_PLAYER_VISIBILITY = "not_applicable"
VALID_SUBJECT_TYPES = {
    "player",
    "referee",
    "cameraman_staff",
    "other_non_player",
    "unknown",
}
NON_PLAYER_SUBJECT_TYPES = {"referee", "cameraman_staff", "other_non_player"}

MANIFEST_FIELDS = [
    "sample_id", "video", "frame_id", "timestamp_sec", "track_id",
    "team_suggested", "team_score", "team_margin", "player_conf",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "player_width", "player_height", "torso_width", "torso_height",
    "torso_sharpness", "player_crop", "torso_crop",
]
ANNOTATION_FIELDS = [
    "sample_id", "subject_type", "jersey_gt", "visibility", "team_gt",
    "identity_group", "reviewer", "notes", "annotated_at",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> float:
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(f"Invalid time: {value}")


def _safe_crop(frame, box: list[float]):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, (x1, y1, x2, y2)
    return crop, (x1, y1, x2, y2)


def _torso_crop(player_crop):
    h, w = player_crop.shape[:2]
    y1, y2 = int(h * 0.15), max(int(h * 0.65), int(h * 0.15) + 1)
    x1, x2 = int(w * 0.10), max(int(w * 0.90), int(w * 0.10) + 1)
    torso = player_crop[y1:y2, x1:x2]
    return torso if torso.size else None


def _sharpness(crop) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _class_name(model: YOLO, cls_id: int) -> str:
    names = model.names
    if isinstance(names, dict):
        value = names.get(cls_id, cls_id)
    else:
        value = names[cls_id] if 0 <= cls_id < len(names) else cls_id
    return str(value).lower()


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _label_kind(value: str) -> str:
    value = str(value or "").strip().lower()
    if not value:
        return "missing"
    if value == "unreadable":
        return "unreadable"
    if value == "not_applicable":
        return "not_applicable"
    if value.isdigit() and 0 <= int(value) <= 99:
        return "number"
    return "invalid"


def _normalize_annotation_row(row: dict[str, str]) -> dict[str, str]:
    """Read legacy annotation CSVs without guessing ambiguous unreadable rows."""
    normalized = {field: str(row.get(field, "") or "") for field in ANNOTATION_FIELDS}
    kind = _label_kind(normalized["jersey_gt"])
    if not normalized["subject_type"] and kind == "number":
        normalized["subject_type"] = "player"
    return normalized


def _split_for_group(group: str, seed: int, train: float, val: float) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train:
        return "train"
    if value < train + val:
        return "val"
    return "test"


def extract(args: argparse.Namespace) -> int:
    video = Path(args.video).expanduser().resolve()
    match_config = Path(args.match_config).expanduser().resolve()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for path in (video, match_config, model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists. Use a new output directory or --overwrite."
        )

    player_dir = output_dir / "crops" / "player"
    torso_dir = output_dir / "crops" / "torso"
    player_dir.mkdir(parents=True, exist_ok=True)
    torso_dir.mkdir(parents=True, exist_ok=True)

    team_analyzer = TeamOnlyAnalyzer(profiling_path=str(match_config))
    model = YOLO(str(model_path))
    model.to(args.device)

    start_sec = _parse_time(args.start)
    end_sec = _parse_time(args.end)
    if end_sec <= start_sec:
        raise ValueError("--end must be greater than --start")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    frame_index = int(round(start_sec * fps))

    sample_every = max(0.05, float(args.sample_every_sec))
    max_per_track = max(1, int(args.max_per_track))
    last_sample_time: dict[int, float] = {}
    track_counts: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []
    processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / max(fps, 1e-6)
            if timestamp > end_sec:
                break

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=float(args.inference_conf),
                imgsz=int(args.imgsz),
                device=args.device,
                max_det=40,
                verbose=False,
            )
            processed += 1

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if _class_name(model, cls_id) != "player":
                        continue
                    if confidence < float(args.player_conf) or box.id is None:
                        continue
                    track_id = int(box.id[0])
                    if track_id < 0 or track_counts[track_id] >= max_per_track:
                        continue
                    previous = last_sample_time.get(track_id)
                    if previous is not None and timestamp - previous < sample_every:
                        continue

                    player, coords = _safe_crop(frame, box.xyxy[0].tolist())
                    if player is None or player.shape[0] < int(args.min_player_height):
                        continue
                    torso = _torso_crop(player)
                    if torso is None or torso.shape[0] < 16 or torso.shape[1] < 12:
                        continue

                    team_result = team_analyzer.score_teams(player)
                    sample_id = f"{video.stem}_f{frame_index:07d}_t{track_id:05d}"
                    player_rel = Path("crops") / "player" / f"{sample_id}.jpg"
                    torso_rel = Path("crops") / "torso" / f"{sample_id}.jpg"
                    if not cv2.imwrite(str(output_dir / player_rel), player):
                        raise RuntimeError(f"Failed to save {player_rel}")
                    if not cv2.imwrite(str(output_dir / torso_rel), torso):
                        raise RuntimeError(f"Failed to save {torso_rel}")

                    x1, y1, x2, y2 = coords
                    rows.append({
                        "sample_id": sample_id,
                        "video": str(video),
                        "frame_id": frame_index,
                        "timestamp_sec": round(timestamp, 4),
                        "track_id": track_id,
                        "team_suggested": team_result.get("best_team") or "",
                        "team_score": team_result.get("best_score", ""),
                        "team_margin": team_result.get("margin", ""),
                        "player_conf": round(confidence, 6),
                        "bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2,
                        "player_width": player.shape[1], "player_height": player.shape[0],
                        "torso_width": torso.shape[1], "torso_height": torso.shape[0],
                        "torso_sharpness": round(_sharpness(torso), 3),
                        "player_crop": player_rel.as_posix(),
                        "torso_crop": torso_rel.as_posix(),
                    })
                    last_sample_time[track_id] = timestamp
                    track_counts[track_id] += 1

            if processed % 250 == 0:
                print(
                    f"[GT EXTRACT] frames={processed} tracks={len(track_counts)} samples={len(rows)}",
                    flush=True,
                )
            frame_index += 1
    finally:
        cap.release()

    rows.sort(key=lambda row: (int(row["track_id"]), float(row["timestamp_sec"])))
    _write_csv(manifest_path, MANIFEST_FIELDS, rows)
    annotations_path = output_dir / "annotations.csv"
    if not annotations_path.exists():
        _write_csv(annotations_path, ANNOTATION_FIELDS, [])

    metadata = {
        "created_at": _utc_now(),
        "video": str(video),
        "match_config": str(match_config),
        "canonical_player_detector": str(model_path),
        "tracker": args.tracker,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "fps": fps,
        "frames_processed": processed,
        "samples": len(rows),
        "tracks": len(track_counts),
        "sample_every_sec": sample_every,
        "max_per_track": max_per_track,
        "imgsz": int(args.imgsz),
        "player_conf": float(args.player_conf),
        "inference_conf": float(args.inference_conf),
        "device": args.device,
        "annotation_policy": {
            "subject_type": sorted(VALID_SUBJECT_TYPES),
            "player_jersey_gt": "0-99 or unreadable; never infer a number from roster membership",
            "player_visibility": sorted(PLAYER_VISIBILITY),
            "non_player": "use jersey_gt=not_applicable and visibility=not_applicable",
            "identity_group": "optional physical-player group used to prevent split leakage",
            "quality": "retain blurry/unreadable player crops so an OCR quality gate can be calibrated later",
        },
    }
    (output_dir / "extraction.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[GT EXTRACT] manifest={manifest_path}")
    print(f"[GT EXTRACT] annotations={annotations_path}")
    print(f"[GT EXTRACT] samples={len(rows)} tracks={len(track_counts)}")
    return 0


def _validate_and_join(
    manifest_path: Path, annotations_path: Path
) -> tuple[list[dict[str, str]], list[str]]:
    manifest = _read_csv(manifest_path)
    raw_annotations = _read_csv(annotations_path) if annotations_path.exists() else []
    annotations = [_normalize_annotation_row(row) for row in raw_annotations]
    errors: list[str] = []
    manifest_by_id: dict[str, dict[str, str]] = {}

    for row in manifest:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            errors.append("manifest contains a row without sample_id")
            continue
        if sample_id in manifest_by_id:
            errors.append(f"duplicate manifest sample_id: {sample_id}")
        manifest_by_id[sample_id] = row

    annotation_by_id: dict[str, dict[str, str]] = {}
    for row in annotations:
        sample_id = row.get("sample_id", "").strip()
        if not sample_id:
            errors.append("annotations contains a row without sample_id")
            continue
        if sample_id not in manifest_by_id:
            errors.append(f"annotation references unknown sample_id: {sample_id}")
            continue
        if sample_id in annotation_by_id:
            errors.append(f"duplicate annotation sample_id: {sample_id}")
        annotation_by_id[sample_id] = row

        subject_type = row.get("subject_type", "").strip().lower()
        kind = _label_kind(row.get("jersey_gt", ""))
        visibility = row.get("visibility", "").strip().lower()

        if kind == "missing" and not subject_type:
            continue
        if not subject_type:
            errors.append(
                f"{sample_id}: subject_type is required; legacy unreadable rows must be reviewed"
            )
            continue
        if subject_type not in VALID_SUBJECT_TYPES:
            errors.append(
                f"{sample_id}: invalid subject_type {subject_type!r}; expected one of {sorted(VALID_SUBJECT_TYPES)}"
            )
            continue

        if subject_type == "player":
            if kind not in {"number", "unreadable"}:
                errors.append(
                    f"{sample_id}: player jersey_gt must be 0-99 or unreadable"
                )
            if visibility not in PLAYER_VISIBILITY:
                errors.append(
                    f"{sample_id}: player visibility must be one of {sorted(PLAYER_VISIBILITY)}"
                )
        else:
            if kind != "not_applicable":
                errors.append(
                    f"{sample_id}: non-player/unknown jersey_gt must be not_applicable"
                )
            if visibility != NON_PLAYER_VISIBILITY:
                errors.append(
                    f"{sample_id}: non-player/unknown visibility must be not_applicable"
                )

    joined: list[dict[str, str]] = []
    track_numbers: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for sample_id, manifest_row in manifest_by_id.items():
        annotation = annotation_by_id.get(sample_id, {})
        row = dict(manifest_row)
        for field in ANNOTATION_FIELDS:
            if field != "sample_id":
                row[field] = annotation.get(field, "")
        row["label_kind"] = _label_kind(row.get("jersey_gt", ""))
        subject_type = row.get("subject_type", "").strip().lower()
        row["ocr_eligible"] = str(
            subject_type == "player" and row["label_kind"] in {"number", "unreadable"}
        ).lower()
        if subject_type in NON_PLAYER_SUBJECT_TYPES:
            row["dataset_role"] = "detector_negative"
        elif subject_type == "player":
            row["dataset_role"] = "ocr"
        elif subject_type == "unknown":
            row["dataset_role"] = "excluded_unknown"
        else:
            row["dataset_role"] = "unreviewed"
        joined.append(row)

        if subject_type == "player" and row["label_kind"] == "number":
            key = (row.get("video", ""), row.get("track_id", ""))
            track_numbers[key].add(str(int(row["jersey_gt"])))

    for (video, track_id), numbers in sorted(track_numbers.items()):
        if len(numbers) > 1:
            errors.append(
                f"track conflict: video={Path(video).name} track_id={track_id} has labels "
                f"{sorted(numbers)}; review for an ID switch or annotation error"
            )
    return joined, errors


def validate(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations else manifest_path.parent / "annotations.csv"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    joined, errors = _validate_and_join(manifest_path, annotations_path)
    kinds = Counter(row["label_kind"] for row in joined)
    subjects = Counter(row.get("subject_type", "") or "missing" for row in joined)
    roles = Counter(row.get("dataset_role", "") for row in joined)
    numbers = Counter(
        str(int(row["jersey_gt"]))
        for row in joined
        if row.get("subject_type") == "player" and row["label_kind"] == "number"
    )
    reviewed_rows = sum(
        1 for row in joined if row.get("dataset_role") in {"ocr", "detector_negative", "excluded_unknown"}
    )
    summary = {
        "manifest_rows": len(joined),
        "reviewed_rows": reviewed_rows,
        "ocr_eligible_rows": roles["ocr"],
        "detector_negative_rows": roles["detector_negative"],
        "unknown_subject_rows": roles["excluded_unknown"],
        "numeric_rows": kinds["number"],
        "unreadable_rows": kinds["unreadable"],
        "not_applicable_rows": kinds["not_applicable"],
        "missing_rows": kinds["missing"],
        "invalid_rows": kinds["invalid"],
        "subject_distribution": dict(sorted(subjects.items())),
        "numeric_distribution": dict(sorted(numbers.items(), key=lambda item: int(item[0]))),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if errors else 0


def export(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations else manifest_path.parent / "annotations.csv"
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output else manifest_path.parent / "ground_truth.csv"
    )
    joined, errors = _validate_and_join(manifest_path, annotations_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit("Ground truth validation failed; export aborted.")

    train, val, test = float(args.train), float(args.val), float(args.test)
    if min(train, val, test) < 0 or abs((train + val + test) - 1.0) > 1e-6:
        raise ValueError("--train + --val + --test must equal 1.0")

    reviewed: list[dict[str, str]] = []
    for row in joined:
        if row.get("dataset_role") not in {"ocr", "detector_negative", "excluded_unknown"}:
            continue
        identity_group = row.get("identity_group", "").strip()
        fallback_group = f"{row.get('video', '')}|track:{row.get('track_id', '')}"
        split_group = identity_group or fallback_group
        out = dict(row)
        out["split_group"] = split_group
        out["split"] = _split_for_group(split_group, int(args.seed), train, val)
        reviewed.append(out)

    if not reviewed:
        raise ValueError("No reviewed annotations to export")

    export_fields = MANIFEST_FIELDS + [
        field for field in ANNOTATION_FIELDS if field != "sample_id"
    ] + ["label_kind", "ocr_eligible", "dataset_role", "split_group", "split"]
    _write_csv(output_path, export_fields, reviewed)
    report = {
        "created_at": _utc_now(),
        "ground_truth": str(output_path),
        "rows": len(reviewed),
        "ocr_eligible_rows": sum(row["ocr_eligible"] == "true" for row in reviewed),
        "detector_negative_rows": sum(row["dataset_role"] == "detector_negative" for row in reviewed),
        "unknown_subject_rows": sum(row["dataset_role"] == "excluded_unknown" for row in reviewed),
        "split_counts": dict(Counter(row["split"] for row in reviewed)),
        "label_counts": dict(Counter(row["label_kind"] for row in reviewed)),
        "subject_counts": dict(Counter(row.get("subject_type", "") for row in reviewed)),
        "dataset_role_counts": dict(Counter(row["dataset_role"] for row in reviewed)),
        "seed": int(args.seed),
        "split_ratios": {"train": train, "val": val, "test": test},
        "grouping": (
            "identity_group when annotated; otherwise video+track_id. Set identity_group "
            "across fragmented tracks of the same physical player before model evaluation."
        ),
        "ocr_policy": (
            "Only dataset_role=ocr / ocr_eligible=true rows are jersey OCR samples. "
            "Non-player detections remain in the ground truth as detector_negative evidence."
        ),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CourtFlux jersey OCR ground-truth extraction and validation"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="Extract tracked player and torso crops")
    p.add_argument("--video", required=True)
    p.add_argument("--match-config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default=str(CANONICAL_MODEL))
    p.add_argument("--start", default="0")
    p.add_argument("--end", default="120")
    p.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    p.add_argument("--tracker", default="botsort.yaml")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--player-conf", type=float, default=0.10)
    p.add_argument("--inference-conf", type=float, default=0.03)
    p.add_argument("--sample-every-sec", type=float, default=1.0)
    p.add_argument("--max-per-track", type=int, default=8)
    p.add_argument("--min-player-height", type=int, default=48)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=extract)

    p = sub.add_parser("validate", help="Validate manifest plus manual annotations")
    p.add_argument("--manifest", required=True)
    p.add_argument("--annotations")
    p.set_defaults(func=validate)

    p = sub.add_parser("export", help="Export reviewed GT with grouped train/val/test split")
    p.add_argument("--manifest", required=True)
    p.add_argument("--annotations")
    p.add_argument("--output")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.set_defaults(func=export)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
