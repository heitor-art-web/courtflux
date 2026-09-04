from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CourtFlux P0.1 one-click ball ground-truth annotator."
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="annotations.jsonl created by prepare_ball_ground_truth.py",
    )
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=900)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No annotation rows found in {path}")
    return rows


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def first_unlabelled(rows: list[dict[str, Any]]) -> int:
    for index, row in enumerate(rows):
        if not isinstance(row.get("visible"), bool):
            return index
    return len(rows)


def fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    if width <= 0 or height <= 0:
        return 1.0
    return min(1.0, max_width / width, max_height / height)


def main() -> None:
    args = parse_args()
    annotation_path = Path(args.annotations).expanduser().resolve()
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)

    rows = load_rows(annotation_path)
    index = first_unlabelled(rows)
    if index >= len(rows):
        print(f"All {len(rows)} frames are already labelled.")
        print("Press B in the annotator after reopening if you want to review the last frame.")
        index = len(rows) - 1

    window = "CourtFlux P0.1 — click ball | N not visible | B back | Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    state: dict[str, Any] = {
        "scale": 1.0,
        "clicked": None,
        "original_width": 0,
        "original_height": 0,
    }

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        scale = float(state.get("scale") or 1.0)
        width = int(state.get("original_width") or 0)
        height = int(state.get("original_height") or 0)
        original_x = max(0, min(width - 1, int(round(x / scale))))
        original_y = max(0, min(height - 1, int(round(y / scale))))
        state["clicked"] = [original_x, original_y]

    cv2.setMouseCallback(window, on_mouse)

    while 0 <= index < len(rows):
        row = rows[index]
        image_value = row.get("image")
        if not image_value:
            raise ValueError(f"Frame {row.get('frame_id')} has no image path")
        image_path = annotation_path.parent / str(image_value)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(image_path)

        height, width = frame.shape[:2]
        scale = fit_scale(width, height, max(320, args.max_width), max(240, args.max_height))
        display = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )

        state["scale"] = scale
        state["clicked"] = None
        state["original_width"] = width
        state["original_height"] = height

        labelled = sum(1 for item in rows if isinstance(item.get("visible"), bool))
        status = "UNLABELLED"
        if row.get("visible") is True:
            status = f"BALL @ {row.get('point')}"
        elif row.get("visible") is False:
            status = "NOT VISIBLE"

        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 72), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.58, display, 0.42, 0)
        cv2.putText(
            display,
            f"{index + 1}/{len(rows)}  labelled={labelled}  frame={row.get('frame_id')}  {status}",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "CLICK ball = save+next   N = not visible   B = back   Q = save+quit",
            (18, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if row.get("visible") is True and isinstance(row.get("point"), list) and len(row["point"]) == 2:
            px = int(round(float(row["point"][0]) * scale))
            py = int(round(float(row["point"][1]) * scale))
            cv2.drawMarker(display, (px, py), (0, 165, 255), cv2.MARKER_CROSS, 22, 2)

        cv2.imshow(window, display)

        while True:
            key = cv2.waitKey(30) & 0xFF
            clicked = state.get("clicked")
            if isinstance(clicked, list) and len(clicked) == 2:
                row["visible"] = True
                row["point"] = [int(clicked[0]), int(clicked[1])]
                # Keep bbox optional. Point truth is enough for the fast P0.1 workflow.
                row["bbox"] = row.get("bbox") if isinstance(row.get("bbox"), list) else None
                save_rows(annotation_path, rows)
                index += 1
                break

            if key in (ord("n"), ord("N")):
                row["visible"] = False
                row["point"] = None
                row["bbox"] = None
                save_rows(annotation_path, rows)
                index += 1
                break
            if key in (ord("b"), ord("B")):
                save_rows(annotation_path, rows)
                index = max(0, index - 1)
                break
            if key in (ord("q"), ord("Q"), 27):
                save_rows(annotation_path, rows)
                cv2.destroyAllWindows()
                labelled = sum(1 for item in rows if isinstance(item.get("visible"), bool))
                print(f"Saved {labelled}/{len(rows)} labelled frames to {annotation_path}")
                return

    save_rows(annotation_path, rows)
    cv2.destroyAllWindows()
    labelled = sum(1 for item in rows if isinstance(item.get("visible"), bool))
    print(f"Completed: {labelled}/{len(rows)} frames labelled in {annotation_path}")


if __name__ == "__main__":
    main()
