from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from jersey_ground_truth import (
    ANNOTATION_FIELDS,
    MANIFEST_FIELDS,
    TRACK_REVIEW_FIELDS,
    _label_kind,
    _normalize_annotation_row,
    _split_for_group,
    _validate_and_join,
)


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    assert _label_kind("") == "missing"
    assert _label_kind("7") == "number"
    assert _label_kind("07") == "number"
    assert _label_kind("99") == "number"
    assert _label_kind("100") == "invalid"
    assert _label_kind("unreadable") == "unreadable"
    assert _label_kind("not_applicable") == "not_applicable"
    assert _split_for_group("player-a", 42, 0.7, 0.15) == _split_for_group(
        "player-a", 42, 0.7, 0.15
    )

    legacy_numeric = _normalize_annotation_row(
        {"sample_id": "legacy", "jersey_gt": "14", "visibility": "clear"}
    )
    assert legacy_numeric["subject_type"] == "player"
    legacy_unreadable = _normalize_annotation_row(
        {"sample_id": "legacy-u", "jersey_gt": "unreadable", "visibility": "unreadable"}
    )
    assert legacy_unreadable["subject_type"] == ""

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manifest = root / "manifest.csv"
        annotations = root / "annotations.csv"
        track_reviews = root / "track_reviews.csv"
        base = {field: "" for field in MANIFEST_FIELDS}
        rows = []
        for index in (1, 2, 3):
            row = dict(base)
            row.update(
                {
                    "sample_id": f"s{index}",
                    "video": "/tmp/game.mp4",
                    "frame_id": str(index),
                    "timestamp_sec": str(index / 25),
                    "track_id": "10" if index < 3 else "20",
                    "player_crop": f"crops/player/s{index}.jpg",
                    "torso_crop": f"crops/torso/s{index}.jpg",
                }
            )
            rows.append(row)
        _write(manifest, MANIFEST_FIELDS, rows)

        a1 = {field: "" for field in ANNOTATION_FIELDS}
        a1.update(
            {
                "sample_id": "s1",
                "subject_type": "player",
                "jersey_gt": "14",
                "visibility": "clear",
                "reviewer": "test",
            }
        )
        a2 = dict(a1)
        a2.update({"sample_id": "s2", "jersey_gt": "14", "visibility": "partial"})
        a3 = {field: "" for field in ANNOTATION_FIELDS}
        a3.update(
            {
                "sample_id": "s3",
                "subject_type": "referee",
                "jersey_gt": "not_applicable",
                "visibility": "not_applicable",
                "reviewer": "test",
            }
        )
        _write(annotations, ANNOTATION_FIELDS, [a1, a2, a3])
        joined, errors = _validate_and_join(manifest, annotations)
        assert len(joined) == 3
        assert not errors
        by_id = {row["sample_id"]: row for row in joined}
        assert by_id["s1"]["ocr_eligible"] == "true"
        assert by_id["s3"]["ocr_eligible"] == "false"
        assert by_id["s3"]["dataset_role"] == "detector_negative"

        a2["jersey_gt"] = "7"
        _write(annotations, ANNOTATION_FIELDS, [a1, a2, a3])
        _, errors = _validate_and_join(manifest, annotations)
        assert any("track conflict" in error for error in errors)

        review = {field: "" for field in TRACK_REVIEW_FIELDS}
        review.update(
            {
                "video": "/tmp/game.mp4",
                "track_id": "10",
                "labels": "7|14",
                "resolution": "confirmed_id_switch",
                "reviewer": "test",
                "reviewed_at": "2026-09-10T12:00:00+00:00",
            }
        )
        _write(track_reviews, TRACK_REVIEW_FIELDS, [review])
        _, errors = _validate_and_join(manifest, annotations, track_reviews)
        assert not errors

        review["labels"] = "6|14"
        _write(track_reviews, TRACK_REVIEW_FIELDS, [review])
        _, errors = _validate_and_join(manifest, annotations, track_reviews)
        assert any("track conflict" in error for error in errors)

        legacy_bad = {
            "sample_id": "s2",
            "jersey_gt": "unreadable",
            "visibility": "unreadable",
            "reviewer": "test",
        }
        legacy_fields = [
            "sample_id", "jersey_gt", "visibility", "team_gt", "identity_group",
            "reviewer", "notes", "annotated_at",
        ]
        _write(annotations, legacy_fields, [legacy_bad])
        _, errors = _validate_and_join(manifest, annotations)
        assert any("subject_type is required" in error for error in errors)

    print("jersey ground truth smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
