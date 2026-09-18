from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _resolve_path(value: Any, *, base: Path) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _video_source_from_run_config(run_dir: Path) -> str | None:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return None
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    # Canonical RunDiagnostics format stores video_source at the top level.
    # Keep nested fallbacks for older/ad-hoc run metadata variants.
    candidates: list[Any] = [payload.get("video_source")]
    for key in ("config", "input", "request"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get("video_source"))
            candidates.append(nested.get("video"))

    for value in candidates:
        source = str(value or "").strip()
        if source:
            return source
    return None


def _resolve_video_from_source(source: str, *, run_dir: Path) -> Path | None:
    """Resolve a run video source without depending on summary.extra.video_path.

    Local paths are tried first. YouTube URLs are resolved through the same downloader
    used by the analytics pipeline, which reuses the deterministic data/<video_id>.mp4
    cache when present.
    """
    text = str(source or "").strip()
    if not text:
        return None

    local_candidates = [
        Path(text).expanduser(),
        (PROJECT_ROOT / text).resolve(),
        (run_dir / text).resolve(),
    ]
    for candidate in local_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    if text.startswith(("https://", "http://")):
        from services.ingestion.youtube_downloader import YouTubeDownloader

        if YouTubeDownloader._video_id_from_url(text):
            downloader = YouTubeDownloader(output_dir=str(PROJECT_ROOT / "data"))
            path = Path(downloader.get_local_video(text)).expanduser()
            if path.exists() and path.is_file():
                return path.resolve()
    return None


def _video_path(summary_path: Path, run_dir: Path) -> tuple[Path, str | None]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates: list[Any] = [summary.get("video_path")]
    extra = summary.get("extra")
    if isinstance(extra, dict):
        candidates.extend(
            [
                extra.get("video_path"),
                extra.get("local_video_path"),
                extra.get("source_video_path"),
            ]
        )
    for value in candidates:
        path = _resolve_path(value, base=summary_path.parent)
        if path is not None and path.exists():
            return path.resolve(), None

    source = _video_source_from_run_config(run_dir)
    if source:
        path = _resolve_video_from_source(source, run_dir=run_dir)
        if path is not None:
            return path, source

    raise FileNotFoundError(
        f"Could not resolve video for {run_dir}. Tried summary video paths and "
        "run_config.json video_source (top-level plus legacy nested variants)."
    )


def _find_state(rows: list[dict[str, Any]], frame_id: int) -> dict[str, Any]:
    for row in rows:
        try:
            current = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            continue
        if current == frame_id:
            return row
    raise ValueError(f"Frame {frame_id} not found in p2_states.jsonl")


def _box_for_track(row: dict[str, Any], track_id: int) -> list[int]:
    for prediction in list(row.get("team_predictions") or []):
        if not isinstance(prediction, dict):
            continue
        try:
            current = int(prediction.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        box = prediction.get("box")
        if current == track_id and isinstance(box, list) and len(box) == 4:
            return [int(round(float(value))) for value in box]
    raise ValueError(
        f"Track {track_id} has no exported box at frame {row.get('frame_id')}"
    )


def _read_frame(video_path: Path, timestamp_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec) * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(
            f"Could not read {video_path} at timestamp {timestamp_sec:.3f}s"
        )
    return frame


def _safe_crop(frame: np.ndarray, box: list[int]) -> tuple[np.ndarray, list[int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid player box after clipping: {box}")
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise ValueError("Player crop is empty")
    return crop, [x1, y1, x2, y2]


def _team_torso_bounds(player: np.ndarray) -> tuple[int, int, int, int]:
    """Exact primary ROI used by TeamOnlyAnalyzer.score_teams()."""
    height, width = player.shape[:2]
    top = max(0, min(height - 1, int(height * 0.22)))
    bottom = max(top + 1, min(height, int(height * 0.58)))
    left = max(0, min(width - 1, int(width * 0.22)))
    right = max(left + 1, min(width, int(width * 0.78)))
    return left, top, right, bottom


def _timestamp(row: dict[str, Any]) -> float:
    for key in ("timestamp_sec", "timestamp", "time_sec"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    raise ValueError(
        f"Frame {row.get('frame_id')} has no timestamp field in p2_states.jsonl"
    )


def _parse_case(raw: str) -> tuple[str, str, int, int]:
    parts = raw.split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--case must be LABEL:RUN_ID:FRAME_ID:TRACK_ID"
        )
    label, run_id, frame_raw, track_raw = parts
    if not label.strip() or not run_id.strip():
        raise argparse.ArgumentTypeError("case label and run id cannot be empty")
    try:
        frame_id = int(frame_raw)
        track_id = int(track_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame and track must be integers") from exc
    return label.strip(), run_id.strip(), frame_id, track_id


def export_case(
    *,
    label: str,
    run_id: str,
    frame_id: int,
    track_id: int,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = PROJECT_ROOT / "evaluation" / "runs" / run_id
    states_path = run_dir / "p2_states.jsonl"
    summary_path = run_dir / "summary.json"
    if not states_path.exists():
        raise FileNotFoundError(states_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    row = _find_state(_load_jsonl(states_path), frame_id)
    box = _box_for_track(row, track_id)
    timestamp_sec = _timestamp(row)
    video_path, video_source = _video_path(summary_path, run_dir)
    frame = _read_frame(video_path, timestamp_sec)
    player, clipped_box = _safe_crop(frame, box)

    left, top, right, bottom = _team_torso_bounds(player)
    torso = player[top:bottom, left:right].copy()
    annotated = player.copy()
    cv2.rectangle(annotated, (left, top), (right - 1, bottom - 1), (0, 255, 255), 2)
    cv2.putText(
        annotated,
        f"{label} f{frame_id} t{track_id} | yellow=TeamOnlyAnalyzer ROI",
        (6, max(18, min(annotated.shape[0] - 6, 22))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
    player_path = output_root / f"{safe_label}_player.jpg"
    torso_path = output_root / f"{safe_label}_torso.jpg"
    if not cv2.imwrite(str(player_path), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Could not write {player_path}")
    if not cv2.imwrite(str(torso_path), torso, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Could not write {torso_path}")

    prediction = None
    for item in list(row.get("team_predictions") or []):
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if current == track_id:
            prediction = item
            break

    return {
        "label": label,
        "run_id": run_id,
        "frame_id": frame_id,
        "track_id": track_id,
        "timestamp_sec": round(timestamp_sec, 6),
        "video_source": video_source,
        "video_path": str(video_path),
        "player_box": clipped_box,
        "torso_roi_in_player": [left, top, right, bottom],
        "player_file": str(player_path),
        "torso_file": str(torso_path),
        "raw_scores": dict((prediction or {}).get("raw_scores") or {}),
        "best_team": (prediction or {}).get("best_team"),
        "evidence_accepted": (prediction or {}).get("evidence_accepted"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export exact player/torso crops for P2 Gate-C colour diagnostics."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=_parse_case,
        metavar="LABEL:RUN_ID:FRAME_ID:TRACK_ID",
        help="Diagnostic case. Repeat for multiple crops.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/crop_audit/gate_c_colour",
        help="Output directory for player/torso JPEGs and manifest.json",
    )
    args = parser.parse_args()

    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    results = [
        export_case(
            label=label,
            run_id=run_id,
            frame_id=frame_id,
            track_id=track_id,
            output_root=output_root,
        )
        for label, run_id, frame_id, track_id in args.case
    ]

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": "courtflux_p2_team_colour_crop_audit_v2", "cases": results}, indent=2),
        encoding="utf-8",
    )

    print(f"[CROP_AUDIT_DIR] {output_root}")
    for result in results:
        print(
            f"[{result['label']}] player={Path(result['player_file']).name} "
            f"torso={Path(result['torso_file']).name} "
            f"frame={result['frame_id']} track={result['track_id']}"
        )
    print(f"[CROP_AUDIT_MANIFEST] {manifest_path}")


if __name__ == "__main__":
    main()
