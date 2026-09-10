from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from jersey_ground_truth import (
    ANNOTATION_FIELDS,
    TRACK_REVIEW_FIELDS,
    _canonical_labels,
    _labels_token,
    _normalize_annotation_row,
    _read_csv,
)


def _atomic_write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACK_REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _parse_track_ids(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids or any(not item.isdigit() for item in ids):
        raise ValueError("--track-ids must be a comma-separated list of numeric track IDs")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record explicit human confirmation of tracker ID switches in jersey GT"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--track-reviews")
    parser.add_argument("--track-ids", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="Manual conflict review confirmed as tracker ID switch")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    annotations_path = (
        Path(args.annotations).expanduser().resolve()
        if args.annotations else manifest_path.parent / "annotations.csv"
    )
    reviews_path = (
        Path(args.track_reviews).expanduser().resolve()
        if args.track_reviews else manifest_path.parent / "track_reviews.csv"
    )

    manifest = _read_csv(manifest_path)
    annotations = [
        _normalize_annotation_row(row)
        for row in _read_csv(annotations_path)
    ]
    manifest_by_id = {row["sample_id"]: row for row in manifest}

    current: dict[tuple[str, str], set[str]] = {}
    for row in annotations:
        if row.get("subject_type", "").strip().lower() != "player":
            continue
        value = row.get("jersey_gt", "").strip()
        if not value.isdigit() or not (0 <= int(value) <= 99):
            continue
        sample = manifest_by_id.get(row.get("sample_id", ""))
        if not sample:
            continue
        key = (sample.get("video", ""), sample.get("track_id", ""))
        current.setdefault(key, set()).add(str(int(value)))

    requested = set(_parse_track_ids(args.track_ids))
    matches: list[tuple[str, str, list[str]]] = []
    for (video, track_id), values in current.items():
        labels = _canonical_labels(list(values))
        if track_id in requested and len(labels) > 1:
            matches.append((video, track_id, labels))

    found = {track_id for _, track_id, _ in matches}
    missing = sorted(requested - found, key=int)
    if missing:
        raise SystemExit(
            "These requested tracks are not currently conflicting: " + ", ".join(missing)
        )

    existing = _read_csv(reviews_path) if reviews_path.exists() and reviews_path.stat().st_size else []
    replace_keys = {(video, track_id) for video, track_id, _ in matches}
    kept = [
        {field: str(row.get(field, "") or "") for field in TRACK_REVIEW_FIELDS}
        for row in existing
        if (str(row.get("video", "")), str(row.get("track_id", ""))) not in replace_keys
    ]

    now = datetime.now(timezone.utc).isoformat()
    additions = [
        {
            "video": video,
            "track_id": track_id,
            "labels": _labels_token(labels),
            "resolution": "confirmed_id_switch",
            "reviewer": args.reviewer.strip(),
            "notes": args.notes.strip(),
            "reviewed_at": now,
        }
        for video, track_id, labels in sorted(matches, key=lambda item: int(item[1]))
    ]
    _atomic_write(reviews_path, kept + additions)

    print(f"track_reviews={reviews_path}")
    for row in additions:
        print(
            f"confirmed track_id={row['track_id']} labels={row['labels']} "
            f"resolution={row['resolution']}"
        )
    print(f"confirmed_tracks={len(additions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
