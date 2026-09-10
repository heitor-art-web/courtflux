from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean, median

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.vision.roboflow_jersey_analyzer import RoboflowJerseyAnalyzer

CANONICAL_MODEL = ROOT / "weights" / "model_v1.0.0.pt"


def _parse_time(value: str) -> float:
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    raise ValueError(value)


def _safe_crop(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _latency_summary(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "count": len(ordered),
        "mean": round(fmean(ordered), 3),
        "median": round(median(ordered), 3),
        "p95": round(ordered[p95_index], 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 Jersey Recognition challenger")
    parser.add_argument("--video", required=True)
    parser.add_argument("--match-config", required=True)
    parser.add_argument("--start", default="0")
    parser.add_argument("--end", default="30")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--player-conf", type=float, default=0.10)
    parser.add_argument("--inference-conf", type=float, default=0.03)
    parser.add_argument(
        "--max-ocr-attempts",
        type=int,
        default=0,
        help="Stop after this many LMM calls; 0 means unlimited. Useful for smoke tests.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = ROOT / video_path
    match_config = Path(args.match_config)
    if not match_config.is_absolute():
        match_config = ROOT / match_config
    if not CANONICAL_MODEL.exists():
        raise FileNotFoundError(CANONICAL_MODEL)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not match_config.exists():
        raise FileNotFoundError(match_config)

    analyzer = RoboflowJerseyAnalyzer(profiling_path=str(match_config))
    model = YOLO(str(CANONICAL_MODEL))
    model.to(args.device)

    start_sec = _parse_time(args.start)
    end_sec = _parse_time(args.end)
    interval = max(1, int(args.interval))
    max_ocr_attempts = max(0, int(args.max_ocr_attempts))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    track_state = defaultdict(
        lambda: {
            "attempts": 0,
            "accepted": 0,
            "last_number": None,
            "consecutive": 0,
            "locked_number": None,
            "locked_team": None,
            "previous_accepted_number": None,
            "consistency_comparisons": 0,
            "consistency_matches": 0,
            "predictions": [],
        }
    )
    frame_index = int(round(start_sec * fps))
    processed = 0
    attempts = 0
    accepted = 0
    latencies_ms: list[float] = []
    consistency_comparisons = 0
    consistency_matches = 0
    stop_requested = False

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
                tracker="botsort.yaml",
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
                    cls_name = str(model.names.get(cls_id, cls_id)).lower()
                    conf = float(box.conf[0])
                    if cls_name != "player" or conf < float(args.player_conf) or box.id is None:
                        continue
                    track_id = int(box.id[0])
                    if track_id < 0:
                        continue
                    crop = _safe_crop(frame, box.xyxy[0].tolist())
                    if crop is None:
                        continue
                    analyzer.observe_track(crop, track_id)
                    state = track_state[track_id]
                    if state["locked_number"] is not None or frame_index % interval != 0:
                        continue

                    team_result = analyzer.score_teams(crop)
                    team = team_result.get("best_team")
                    prediction = analyzer.predict_jersey(track_id, team)
                    attempts += 1
                    state["attempts"] += 1
                    latency_ms = prediction.get("latency_ms")
                    if latency_ms is not None:
                        latencies_ms.append(float(latency_ms))
                    record = {
                        "frame_id": frame_index,
                        "timestamp": round(timestamp, 4),
                        "team": team,
                        "accepted": bool(prediction.get("accepted")),
                        "number": prediction.get("candidate_number"),
                        "reason": prediction.get("rejection_reason"),
                        "latency_ms": latency_ms,
                    }
                    state["predictions"].append(record)
                    if prediction.get("accepted"):
                        accepted += 1
                        state["accepted"] += 1
                        number = str(prediction.get("candidate_number"))

                        previous_accepted = state["previous_accepted_number"]
                        if previous_accepted is not None:
                            state["consistency_comparisons"] += 1
                            consistency_comparisons += 1
                            if number == previous_accepted:
                                state["consistency_matches"] += 1
                                consistency_matches += 1
                        state["previous_accepted_number"] = number

                        if number == state["last_number"]:
                            state["consecutive"] += 1
                        else:
                            state["last_number"] = number
                            state["consecutive"] = 1

                        if state["consecutive"] >= analyzer.identity_confirmations_required:
                            state["locked_number"] = number
                            state["locked_team"] = team
                    else:
                        state["last_number"] = None
                        state["consecutive"] = 0

                    if max_ocr_attempts and attempts >= max_ocr_attempts:
                        stop_requested = True
                        break

            if processed % 100 == 0:
                locks = sum(1 for item in track_state.values() if item["locked_number"] is not None)
                print(
                    f"[PROGRESS] frames={processed} attempts={attempts} accepted={accepted} locks={locks}",
                    flush=True,
                )
            if stop_requested:
                break
            frame_index += 1
    finally:
        cap.release()

    locks = {
        str(track_id): {
            "team": state["locked_team"],
            "number": state["locked_number"],
            "attempts": state["attempts"],
            "accepted": state["accepted"],
            "accepted_prediction_consistency_comparisons": state["consistency_comparisons"],
            "accepted_prediction_consistency_matches": state["consistency_matches"],
            "accepted_prediction_consistency_rate": (
                state["consistency_matches"] / state["consistency_comparisons"]
                if state["consistency_comparisons"]
                else None
            ),
            "predictions": state["predictions"],
        }
        for track_id, state in track_state.items()
        if state["locked_number"] is not None
    }
    payload = {
        "created_at": datetime.now().isoformat(),
        "benchmark_scope": "jersey_recognition_without_ground_truth_accuracy_claim",
        "canonical_player_detector": str(CANONICAL_MODEL.relative_to(ROOT)),
        "tracker": "botsort.yaml",
        "jersey_backend": "roboflow_multimodal",
        "jersey_model_id": analyzer.model_id,
        "jersey_api_url": analyzer.api_url,
        "identity_confirmation_rule": f"{analyzer.identity_confirmations_required}_consecutive_predictions",
        "video": str(video_path),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "fps": fps,
        "frames_processed": processed,
        "ocr_attempts": attempts,
        "ocr_accepted": accepted,
        "ocr_acceptance_rate": (accepted / attempts if attempts else None),
        "accepted_prediction_consistency": {
            "definition": (
                "same-number agreement between successive accepted OCR predictions on the "
                "same track; this is a stability metric, not jersey-number accuracy"
            ),
            "comparisons": consistency_comparisons,
            "matches": consistency_matches,
            "rate": (
                consistency_matches / consistency_comparisons
                if consistency_comparisons
                else None
            ),
        },
        "ocr_latency_ms": _latency_summary(latencies_ms),
        "identity_locks": len(locks),
        "locked_tracks": locks,
        "limitations": [
            "No jersey-number ground truth is used by this runner; identity accuracy is not claimed.",
            "Team labels may remain unknown when the supplied match config has no valid team context.",
            "Accepted-prediction consistency measures stability only and must not be interpreted as accuracy.",
            "The Roboflow VLM is evaluated on CourtFlux player/torso crops before production promotion.",
        ],
    }
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "evaluation" / "p1_jersey_reports" / f"jersey_challenger_{stamp}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[P1 JERSEY REPORT] {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
