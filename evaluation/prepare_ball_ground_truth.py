from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames and create a CourtFlux P0.1 ball ground-truth JSONL template."
    )
    parser.add_argument("--video", required=True, help="Local video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=30.0, help="End time in seconds")
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Extract every Nth source frame. Use 1 for full-frame truth, 2/3 for an initial detector sample.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    every = max(1, int(args.every))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        raise RuntimeError("Video FPS is unavailable")

    start_frame = max(0, int(round(args.start * fps)))
    end_frame = max(start_frame + 1, int(round(args.end * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    annotations_path = output_dir / "annotations.jsonl"
    metadata_path = output_dir / "metadata.json"
    rows: list[dict] = []
    source_frame = start_frame
    extracted = 0

    while source_frame < end_frame:
        ok, frame = capture.read()
        if not ok:
            break

        if (source_frame - start_frame) % every == 0:
            timestamp_sec = source_frame / fps
            file_name = f"frame_{source_frame:06d}_t{timestamp_sec:09.3f}.jpg"
            file_path = frames_dir / file_name
            cv2.imwrite(
                str(file_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(int(args.jpeg_quality), 100))],
            )
            rows.append(
                {
                    "frame_id": source_frame,
                    "timestamp_sec": round(timestamp_sec, 6),
                    "image": f"frames/{file_name}",
                    "visible": None,
                    "bbox": None,
                    "occlusion": None,
                    "court_region": None,
                    "camera": None,
                    "event_context": None,
                    "notes": "",
                }
            )
            extracted += 1

        source_frame += 1

    capture.release()

    with annotations_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "video": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "start_sec": float(args.start),
        "end_sec": float(args.end),
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "sample_every_n_frames": every,
        "frames_extracted": extracted,
        "annotation_file": str(annotations_path),
        "bbox_format": "xyxy pixels in original source frame",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Prepared {extracted} frames in {output_dir}")
    print(f"Annotate: {annotations_path}")
    print("Set visible=true + bbox=[x1,y1,x2,y2] when the ball is visible.")
    print("Set visible=false + bbox=null only when the real ball is genuinely not visible.")


if __name__ == "__main__":
    main()
