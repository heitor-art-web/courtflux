from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODES = ("off", "verify", "zoom", "full")
SUMMARY_RE = re.compile(r"^\[SUMMARY\]\s+(.+)$", re.MULTILINE)


def _nested(data: dict[str, Any], *keys: str, default=None):
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _run_mode(args: argparse.Namespace, mode: str) -> tuple[Path, dict[str, Any]]:
    env = os.environ.copy()
    env.update(
        {
            "BASKETBALL_SINGLE_FORWARD": "1",
            "BASKETBALL_SMART_BALL_MODE": mode,
            "BASKETBALL_BALL_CROP_IMGSZ": str(args.crop_imgsz),
            "BASKETBALL_INFERENCE_CONF": str(args.inference_conf),
            "BASKETBALL_BALL_CONF": str(args.ball_conf),
        }
    )

    command = [
        sys.executable,
        "-m",
        "services.analytics.analyzer",
        "--video",
        args.video,
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--model",
        args.model,
        "--conf",
        str(args.player_conf),
        "--imgsz",
        str(args.imgsz),
        "--device",
        args.device,
        "--tracker",
        args.tracker,
        "--match-config",
        args.match_config,
        "--detection-only",
        "--jersey-interval",
        str(args.jersey_interval),
        "--diagnostics-batch-size",
        "100",
    ]
    if not args.persist:
        command.append("--no-persist")

    print(f"\n=== Smart Ball mode: {mode} ===", flush=True)
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"mode={mode} failed with exit code {completed.returncode}")

    matches = SUMMARY_RE.findall(completed.stdout)
    if not matches:
        raise RuntimeError(f"mode={mode} completed without a [SUMMARY] path")
    summary_path = Path(matches[-1].strip()).expanduser()
    if not summary_path.is_absolute():
        summary_path = (PROJECT_ROOT / summary_path).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary_path, summary


def _row(mode: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "summary": str(path),
        "metric_eligible_rate": _nested(summary, "metric_scope", "metric_eligible_rate"),
        "players_final_per_gameplay_frame": _nested(summary, "players", "final_per_gameplay_frame"),
        "team_classification_rate": _nested(summary, "players", "team_classification_rate"),
        "jersey_identification_rate": _nested(summary, "players", "jersey_identification_rate"),
        "ball_raw_candidate_frame_rate": _nested(summary, "ball", "raw_candidate_frame_rate"),
        "ball_direct_frame_rate": _nested(summary, "ball", "selected_detection_frame_rate"),
        "ball_predicted_hold_frame_rate": _nested(summary, "ball", "predicted_hold_frame_rate"),
        "ball_effective_trajectory_frame_rate": _nested(summary, "ball", "effective_trajectory_frame_rate"),
        "ball_missing_frame_rate": _nested(summary, "ball", "missing_frame_rate"),
        "ball_multi_candidate_frame_rate": _nested(summary, "ball", "multi_candidate_frame_rate"),
        "ball_association_rejection_rate": _nested(
            summary,
            "ball",
            "trajectory",
            "association_candidate_rejection_rate",
        ),
        "smart_ball_ms": _nested(summary, "detector_timing", "smart_ball", "avg_ms", default=0.0),
        "yolo_full_frame_ms": _nested(
            summary,
            "detector_timing",
            "yolo_predict_full_frame",
            "avg_ms",
        ),
        "team_ms": _nested(summary, "detector_timing", "team_classification", "avg_ms"),
        "jersey_ms": _nested(summary, "detector_timing", "jersey_recognition", "avg_ms"),
        "detector_ms": _nested(summary, "performance", "stages", "Detector", "avg_ms"),
        "analysis_fps": _nested(summary, "performance", "analysis_fps"),
        "wall_time_sec": _nested(summary, "performance", "wall_time_sec"),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "mode",
        "direct%",
        "pred%",
        "effective%",
        "missing%",
        "smart ms",
        "detector ms",
        "fps",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        values = [
            row["mode"],
            row["ball_direct_frame_rate"],
            row["ball_predicted_hold_frame_rate"],
            row["ball_effective_trajectory_frame_rate"],
            row["ball_missing_frame_rate"],
            row["smart_ball_ms"],
            row["detector_ms"],
            row["analysis_fps"],
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="CourtFlux Smart Ball four-mode ablation")
    parser.add_argument("--video", default="data/OwiQgAMqyUs.mp4")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=60.0)
    parser.add_argument("--model", default="weights/model_v1.1.0.pt")
    parser.add_argument("--match-config", default="data/match_config.json")
    parser.add_argument("--player-conf", type=float, default=0.40)
    parser.add_argument("--inference-conf", type=float, default=0.03)
    parser.add_argument("--ball-conf", type=float, default=0.075)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--crop-imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--tracker", default="botsort.yaml", choices=["botsort.yaml", "bytetrack.yaml"])
    parser.add_argument("--jersey-interval", type=int, default=24)
    parser.add_argument("--persist", action="store_true", help="Persist SQLite results for every ablation run")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        path, summary = _run_mode(args, mode)
        rows.append(_row(mode, path, summary))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "evaluation" / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"smart_ball_{timestamp}.json"
    md_path = output_dir / f"smart_ball_{timestamp}.md"
    payload = {
        "contract": {
            "video": args.video,
            "start": args.start,
            "end": args.end,
            "model": args.model,
            "player_conf": args.player_conf,
            "inference_conf": args.inference_conf,
            "ball_conf": args.ball_conf,
            "imgsz": args.imgsz,
            "crop_imgsz": args.crop_imgsz,
            "device": args.device,
            "tracker": args.tracker,
            "jersey_enabled": True,
            "jersey_interval": args.jersey_interval,
            "single_forward": True,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(rows), encoding="utf-8")

    print("\n=== Smart Ball ablation comparison ===")
    print(_markdown(rows), end="")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
