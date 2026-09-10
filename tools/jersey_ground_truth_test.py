from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from jersey_ground_truth import (
    ANNOTATION_FIELDS,
    MANIFEST_FIELDS,
    _label_kind,
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
    assert _split_for_group("player-a", 42, 0.7, 0.15) == _split_for_group(
        "player-a", 42, 0.7, 0.15
    )

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manifest = root / "manifest.csv"
        annotations = root / "annotations.csv"
        base = {field: "" for field in MANIFEST_FIELDS}
        rows = []
        for index in (1, 2):
            row = dict(base)
            row.update(
                {
                    "sample_id": f"s{index}",
                    "video": "/tmp/game.mp4",
                    "frame_id": str(index),
                    "timestamp_sec": str(index / 25),
                    "track_id": "10",
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
                "jersey_gt": "14",
                "visibility": "clear",
                "reviewer": "test",
            }
        )
        a2 = dict(a1)
        a2.update({"sample_id": "s2", "jersey_gt": "14", "visibility": "partial"})
        _write(annotations, ANNOTATION_FIELDS, [a1, a2])
        joined, errors = _validate_and_join(manifest, annotations)
        assert len(joined) == 2
        assert not errors

        a2["jersey_gt"] = "7"
        _write(annotations, ANNOTATION_FIELDS, [a1, a2])
        _, errors = _validate_and_join(manifest, annotations)
        assert any("track conflict" in error for error in errors)

    print("jersey ground truth smoke tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
