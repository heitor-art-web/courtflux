from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]


def find_annotation(dataset_root: Path) -> Path:
    candidates = sorted(dataset_root.glob("*/_annotations.coco.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No */_annotations.coco.json found under {dataset_root}"
        )
    # Prefer test, then valid, then train for a compact human inspection sample.
    order = {"test": 0, "valid": 1, "train": 2}
    return min(candidates, key=lambda p: order.get(p.parent.name, 99))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the basketball-court-detection-2 COCO keypoint IDs on the image "
            "with the largest number of visible/occluded labelled landmarks."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="data/external/roboflow/basketball_court_detection_v19",
    )
    parser.add_argument(
        "--output",
        default="evaluation/court_keypoint_schema_v19.jpg",
    )
    args = parser.parse_args()

    dataset_root = (ROOT / args.dataset_root).resolve()
    annotation_path = find_annotation(dataset_root)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))

    categories = payload.get("categories") or []
    if not categories:
        raise RuntimeError("COCO annotations contain no categories")
    category = categories[0]
    labels = [str(v) for v in category.get("keypoints") or []]
    if not labels:
        raise RuntimeError("COCO category contains no keypoint names")

    images = {int(image["id"]): image for image in payload.get("images") or []}
    best = None
    best_count = -1
    for annotation in payload.get("annotations") or []:
        values = annotation.get("keypoints") or []
        if len(values) != len(labels) * 3:
            continue
        labelled = sum(1 for i in range(len(labels)) if int(values[3 * i + 2]) > 0)
        if labelled > best_count:
            best_count = labelled
            best = annotation

    if best is None:
        raise RuntimeError("No annotation with the expected keypoint schema found")

    image_meta = images.get(int(best["image_id"]))
    if image_meta is None:
        raise RuntimeError("Selected annotation references an unknown image")
    image_path = annotation_path.parent / str(image_meta["file_name"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    values = best["keypoints"]
    rows = []
    for index, label in enumerate(labels):
        x = float(values[3 * index])
        y = float(values[3 * index + 1])
        visibility = int(values[3 * index + 2])
        rows.append(
            {
                "index": index,
                "label": label,
                "x": round(x, 3),
                "y": round(y, 3),
                "visibility": visibility,
            }
        )
        if visibility <= 0:
            continue
        center = (int(round(x)), int(round(y)))
        cv2.circle(image, center, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to write {output}")

    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "annotation_path": str(annotation_path),
                "image_path": str(image_path),
                "visible_or_occluded_keypoints": best_count,
                "labels": labels,
                "points": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"schema labels ({len(labels)}): {labels}")
    print(f"selected image: {image_path}")
    print(f"labelled landmarks: {best_count}/{len(labels)}")
    print(f"rendered: {output}")
    print(f"metadata: {json_path}")


if __name__ == "__main__":
    main()
