from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.player_tracking_quality import aggregate_reports, evaluate_frames
from services.gameplay_filter import GameplayFilter
from services.vision.rfdetr_botsort_adapter import RFDETRBoTSORTAdapter
from services.vision.rfdetr_player_detector import RFDETRPlayerDetector

CONTRACT = ROOT / "evaluation" / "p1_tracking_baseline_contract.json"


def _run_video(video: dict, *, inference_conf: float, player_conf: float) -> dict:
    path = (ROOT / video["path"]).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, float(video["start_sec"]) * 1000.0)

    detector = RFDETRPlayerDetector(
        inference_threshold=inference_conf,
        player_threshold=player_conf,
    )
    tracker = RFDETRBoTSORTAdapter("botsort.yaml")
    gameplay_filter = GameplayFilter(
        min_court_ratio=0.15,
        min_motion_ratio=0.003,
        max_grace_frames=15,
        processing_scale=0.5,
    )

    rows = []
    frame_id = 0
    detector_ms = []
    start_wall = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if timestamp > float(video["end_sec"]):
                break

            valid_gameplay = gameplay_filter.is_gameplay(frame)
            players = []
            detected_count = 0
            timing = {}
            if valid_gameplay:
                t0 = time.perf_counter()
                detections = detector.detect(frame)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                detector_ms.append(elapsed_ms)
                timing["rfdetr_players"] = elapsed_ms
                detected_count = len(detections)
                tracked = tracker.update(detections, frame)
                frame_height = int(frame.shape[0])
                for player in tracked:
                    players.append(
                        {
                            "track_id": int(player.track_id),
                            "box": [round(float(v), 3) for v in player.box],
                            "confidence": round(float(player.confidence), 6),
                            "team": "UNKNOWN",
                            "shirt_num": None,
                            "player_name": None,
                            "court_xy": None,
                            "court_xy_valid": False,
                            "homography_source": None,
                            "frame_height": frame_height,
                        }
                    )

            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": timestamp,
                    "valid_gameplay": bool(valid_gameplay),
                    "accepted_player_candidates": detected_count,
                    "players": players,
                    "timing_ms": timing,
                }
            )
            frame_id += 1
    finally:
        cap.release()

    report = evaluate_frames(rows)
    report["benchmark_context"] = {
        "label": video["label"],
        "video": video["path"],
        "detector": "RF-DETR Small COCO person",
        "tracker": "botsort.yaml",
        "team_context": "unknown_match",
        "team_metrics_in_quality_gate": False,
    }
    report["challenger_runtime"] = {
        "video_fps": fps,
        "wall_time_sec": round(time.perf_counter() - start_wall, 3),
        "player_detector_ms_per_gameplay_frame_mean": (
            round(sum(detector_ms) / len(detector_ms), 6) if detector_ms else None
        ),
    }
    return {"rows": rows, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CourtFlux P1.1 RF-DETR player detector challenger")
    parser.add_argument("--video", choices=["V1", "V2", "V3", "V4", "V5"], default=None)
    parser.add_argument("--inference-conf", type=float, default=0.03)
    parser.add_argument("--player-conf", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    videos = [v for v in contract["videos"] if args.video is None or v["label"] == args.video]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = ROOT / "evaluation" / "p1_rfdetr_runs" / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    reports = []
    runs = []
    for video in videos:
        print(f"\n=== RF-DETR {video['label']} {video['path']} {video['start_sec']}–{video['end_sec']}s ===", flush=True)
        result = _run_video(video, inference_conf=args.inference_conf, player_conf=args.player_conf)
        video_dir = run_root / video["label"]
        video_dir.mkdir(parents=True, exist_ok=True)
        frames_path = video_dir / "frames.jsonl"
        with frames_path.open("w", encoding="utf-8") as handle:
            for row in result["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = result["report"]
        reports.append(report)
        runs.append({"label": video["label"], "frames_file": str(frames_path), "video": video})
        print(
            f"{video['label']}: eligible players/frame="
            f"{report['frames']['analytics_eligible']['players_per_frame_mean']} "
            f"assignment={report['detector_tracker_bridge']['tracker_assignment_rate_pct']}%",
            flush=True,
        )

    payload = {
        "created_at": datetime.now().isoformat(),
        "schema_version": 3,
        "scope": "P1.1_rfdetr_challenger",
        "challenger": {
            "detector": "RF-DETR Small pretrained COCO person",
            "inference_confidence": args.inference_conf,
            "player_confidence": args.player_conf,
            "tracker": "botsort.yaml",
            "reid": False,
            "canonical_yolo_baseline_unchanged": "weights/model_v1.0.0.pt",
        },
        "runs": runs,
        "reports": reports,
        "aggregate": aggregate_reports(reports),
    }
    output = args.output or (ROOT / "evaluation" / "p1_tracking_reports" / f"rfdetr_challenger_{args.video or 'v1_v5'}_{stamp}.json")
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[P1.1 RF-DETR REPORT] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
