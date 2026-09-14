from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from evaluation.p1_jersey_probe import PROJECT_ROOT, _json_lines, _record, _resolve_path
from services.vision.p1_jersey_analyzer import PerformanceJerseyAnalyzer


def _resolve_device(requested: str) -> str:
    value = requested.strip().lower()
    if value != "auto":
        if value == "mps" and torch.backends.mps.is_available():
            return "mps"
        if value == "cuda" and torch.cuda.is_available():
            return "cuda"
        if value == "cpu":
            return "cpu"
        raise RuntimeError(f"Requested device {requested!r} is unavailable")
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_player_crop(frame: np.ndarray, box: list[int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid player box after clipping: {box}")
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Player crop is empty")
    return crop


def _variant(player: np.ndarray, y0: float, y1: float, x0: float, x1: float) -> np.ndarray:
    height, width = player.shape[:2]
    top = max(0, min(height - 1, int(round(height * y0))))
    bottom = max(top + 1, min(height, int(round(height * y1))))
    left = max(0, min(width - 1, int(round(width * x0))))
    right = max(left + 1, min(width, int(round(width * x1))))
    return player[top:bottom, left:right]


def _crop_variants(player: np.ndarray) -> dict[str, np.ndarray]:
    return {
        # Exact runtime geometry used by PerformanceJerseyAnalyzer.observe_track().
        "runtime_torso_15_65_10_90": _variant(player, 0.15, 0.65, 0.10, 0.90),
        # Slightly wider/higher crop in case shoulders/side angle move the digits.
        "wide_upper_08_62_03_97": _variant(player, 0.08, 0.62, 0.03, 0.97),
        # Lower torso catches numbers printed lower on the back/front.
        "wide_mid_15_78_03_97": _variant(player, 0.15, 0.78, 0.03, 0.97),
        # Control: whole player box. This is not proposed for production; it tells us
        # whether the checkpoint depends on context that the runtime torso crop removes.
        "full_player_control": player.copy(),
    }


def _top(probabilities: list[float], limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    return [
        {"digit": int(index), "probability": round(float(value), 6)}
        for index, value in ordered[:limit]
    ]


def _expected_score(expected: str | None, tens: list[float], ones: list[float]) -> dict[str, Any] | None:
    if expected is None:
        return None
    canonical = str(int(str(expected).strip()))
    if len(canonical) == 1:
        tens_digit, ones_digit = 0, int(canonical)
    elif len(canonical) == 2:
        tens_digit, ones_digit = int(canonical[0]), int(canonical[1])
    else:
        raise ValueError("Expected jersey must be between 0 and 99")
    tens_probability = float(tens[tens_digit]) if tens_digit < len(tens) else 0.0
    ones_probability = float(ones[ones_digit]) if ones_digit < len(ones) else 0.0
    return {
        "number": canonical,
        "tens_digit": tens_digit,
        "ones_digit": ones_digit,
        "tens_probability": round(tens_probability, 6),
        "ones_probability": round(ones_probability, 6),
        "pair_probability": round(tens_probability * ones_probability, 8),
    }


def _infer_variant(
    analyzer: PerformanceJerseyAnalyzer,
    image: np.ndarray,
    *,
    expected: str | None,
) -> dict[str, Any]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = analyzer.transform(Image.fromarray(rgb)).unsqueeze(0).to(analyzer.device)
    with torch.inference_mode():
        tens_logits, ones_logits = analyzer.number_model(tensor)
        tens = torch.softmax(tens_logits, dim=1)[0].detach().cpu().tolist()
        ones = torch.softmax(ones_logits, dim=1)[0].detach().cpu().tolist()
    raw = dict(analyzer._raw_prediction(tens, ones))
    return {
        "raw_canonical": raw.get("raw_canonical"),
        "raw_probability": raw.get("raw_probability"),
        "tens_top5": _top([float(value) for value in tens]),
        "ones_top5": _top([float(value) for value in ones]),
        "tens_probabilities": [round(float(value), 8) for value in tens],
        "ones_probabilities": [round(float(value), 8) for value in ones],
        "expected": _expected_score(expected, tens, ones),
    }


def _box_for_track(frame_row: dict[str, Any], track_id: int) -> list[int] | None:
    for item in list(frame_row.get("team_predictions") or []):
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        box = item.get("box")
        if current == track_id and isinstance(box, list) and len(box) == 4:
            return [int(value) for value in box]
    return None


def _select_from_summary(
    summary_path: Path,
    *,
    track_id: int,
    frame_id: int | None,
) -> tuple[Path, float, list[int], int, str | None, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames_path = _resolve_path(summary.get("frames_file"), base=summary_path.parent)
    if frames_path is None or not frames_path.exists():
        raise FileNotFoundError("Could not resolve frames.jsonl from summary")
    frame_rows = _json_lines(frames_path)

    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for frame_row in frame_rows:
        current_frame = int(frame_row.get("frame_id", -1) or -1)
        if frame_id is not None and current_frame != frame_id:
            continue
        box = _box_for_track(frame_row, track_id)
        if box is None:
            continue
        matching_prediction = None
        for prediction in list(frame_row.get("jersey_predictions") or []):
            if not isinstance(prediction, dict):
                continue
            try:
                current_track = int(prediction.get("track_id", -1) or -1)
            except (TypeError, ValueError):
                continue
            if current_track == track_id:
                matching_prediction = prediction
                break
        if matching_prediction is None and frame_id is None:
            continue
        quality = 0.0
        record: dict[str, Any] = {}
        if matching_prediction is not None:
            record = _record(frame_row, matching_prediction)
            quality = float(record.get("mean_crop_quality", 0.0) or 0.0)
        candidates.append((quality, frame_row, record))

    if not candidates:
        suffix = f" at frame {frame_id}" if frame_id is not None else ""
        raise ValueError(f"Track {track_id} not found in a Jersey attempt{suffix}")
    _, selected_row, selected_record = max(candidates, key=lambda item: item[0])
    selected_frame_id = int(selected_row.get("frame_id", -1) or -1)
    timestamp = float(selected_row.get("timestamp_sec", 0.0) or 0.0)
    box = _box_for_track(selected_row, track_id)
    if box is None:
        raise ValueError("Selected frame has no player box")

    video_path = _resolve_path(
        (
            (summary.get("extra") or {}).get("video_path")
            if isinstance(summary.get("extra"), dict)
            else None
        ),
        base=summary_path.parent,
    )
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Could not resolve local video from summary")
    team = selected_record.get("team") if selected_record else None
    return video_path, timestamp, box, selected_frame_id, team, selected_record


def _read_frame(video_path: Path, timestamp: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(f"Could not read video frame at {timestamp:.3f}s")
    return frame


def _save_contact_sheet(
    path: Path,
    player: np.ndarray,
    variants: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
    *,
    title: str,
) -> None:
    tile_w, tile_h = 360, 330
    image_h = 220
    names = ["player_box", *variants.keys()]
    columns = 3
    rows = 2
    sheet = np.full((75 + rows * tile_h, columns * tile_w, 3), 18, dtype=np.uint8)
    cv2.putText(sheet, title[:150], (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(
        sheet,
        "Diagnostic only: crop/checkpoint comparison; roster never selects a number",
        (14, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    images = {"player_box": player, **variants}
    for index, name in enumerate(names[: rows * columns]):
        row, column = divmod(index, columns)
        x0, y0 = column * tile_w, 75 + row * tile_h
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (70, 70, 70), 1)
        image = images[name]
        h, w = image.shape[:2]
        scale = min((tile_w - 16) / max(w, 1), image_h / max(h, 1))
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
        rh, rw = resized.shape[:2]
        x = x0 + (tile_w - rw) // 2
        y = y0 + 8 + (image_h - rh) // 2
        sheet[y:y + rh, x:x + rw] = resized
        cv2.putText(sheet, name[:42], (x0 + 10, y0 + image_h + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (235, 235, 235), 1, cv2.LINE_AA)
        if name == "player_box":
            continue
        result = results[name]
        expected = result.get("expected") or {}
        text1 = f"raw={result.get('raw_canonical')} p={float(result.get('raw_probability') or 0):.3f}"
        text2 = "T " + " ".join(
            f"{item['digit']}:{item['probability']:.2f}" for item in result.get("tens_top5", [])[:3]
        )
        text3 = "O " + " ".join(
            f"{item['digit']}:{item['probability']:.2f}" for item in result.get("ones_top5", [])[:3]
        )
        text4 = (
            f"expected pair={float(expected.get('pair_probability', 0.0)):.3f}"
            if expected
            else "expected=n/a"
        )
        for line_index, text in enumerate([text1, text2, text3, text4]):
            cv2.putText(
                sheet,
                text[:52],
                (x0 + 10, y0 + image_h + 55 + line_index * 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run jersey_number_recon.pt directly on one tracked player using several crop "
            "geometries. This bypasses temporal consensus, roster decoding and identity gates."
        )
    )
    parser.add_argument("--summary", required=True, help="Completed run summary.json")
    parser.add_argument("--track-id", required=True, type=int, help="Tracked player id")
    parser.add_argument("--frame-id", type=int, default=None, help="Exact frame; otherwise highest-quality Jersey attempt for the track")
    parser.add_argument("--expected", default=None, help="Visual jersey number, e.g. 23")
    parser.add_argument("--team", default=None, help="Optional team override for diagnostics")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--match-config", default=str(PROJECT_ROOT / "data" / "match_config.json"))
    parser.add_argument("--model", default=str(PROJECT_ROOT / "weights" / "jersey_number_recon.pt"))
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    summary_path = Path(args.summary).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    video_path, timestamp, box, frame_id, detected_team, runtime_record = _select_from_summary(
        summary_path,
        track_id=args.track_id,
        frame_id=args.frame_id,
    )
    team = args.team or detected_team
    frame = _read_frame(video_path, timestamp)
    player = _safe_player_crop(frame, box)
    variants = _crop_variants(player)
    device = _resolve_device(args.device)

    analyzer = PerformanceJerseyAnalyzer(
        profiling_path=str(Path(args.match_config).resolve()),
        device=device,
        model_path=str(Path(args.model).resolve()),
        sequence_length=8,
    )
    results = {
        name: _infer_variant(analyzer, crop, expected=args.expected)
        for name, crop in variants.items()
    }

    crops_root = output_root / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(crops_root / "player_box.jpg"), player)
    for name, crop in variants.items():
        cv2.imwrite(str(crops_root / f"{name}.jpg"), crop)

    valid_roster_numbers = analyzer._valid_roster_numbers(team) if team else []
    payload = {
        "diagnostic_only": True,
        "roster_policy": "validator_only",
        "roster_used_for_decoding": False,
        "summary": str(summary_path),
        "video_path": str(video_path),
        "frame_id": frame_id,
        "timestamp_sec": timestamp,
        "track_id": args.track_id,
        "box": box,
        "team": team,
        "expected_visual_number": args.expected,
        "device": device,
        "runtime_attempt": {
            "candidate_number": runtime_record.get("candidate_number"),
            "accepted": runtime_record.get("accepted"),
            "rejection_reason": runtime_record.get("rejection_reason"),
            "candidate_posterior": runtime_record.get("candidate_posterior"),
            "candidate_margin": runtime_record.get("candidate_margin"),
            "tens_top3": runtime_record.get("tens_top3"),
            "ones_top3": runtime_record.get("ones_top3"),
        },
        "valid_roster_numbers_for_team": valid_roster_numbers,
        "variants": results,
    }
    output_json = output_root / "crop_probe.json"
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _save_contact_sheet(
        output_root / "crop_probe_contact_sheet.jpg",
        player,
        variants,
        results,
        title=(
            f"track {args.track_id} frame {frame_id} {timestamp:.2f}s team={team or 'UNKNOWN'} "
            f"expected={args.expected or '?'}"
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
