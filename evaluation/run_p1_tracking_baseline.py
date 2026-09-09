from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.player_tracking_quality import aggregate_reports, evaluate_run

DEFAULT_CONTRACT = ROOT / "evaluation" / "p1_tracking_baseline_contract.json"
RUN_DIR_RE = re.compile(r"^\[RUN_DIR\]\s+(.+?)\s*$")


def _run_one(video: dict, baseline: dict, device: str) -> Path:
    command = [
        sys.executable,
        "-m",
        "services.analytics.analyzer",
        "--video",
        str(video["path"]),
        "--start",
        str(video["start_sec"]),
        "--end",
        str(video["end_sec"]),
        "--model",
        str(baseline["model"]),
        "--conf",
        str(baseline["player_confidence"]),
        "--imgsz",
        str(baseline["imgsz"]),
        "--device",
        device,
        "--tracker",
        str(baseline["tracker"]),
        "--detection-only",
        "--no-jersey",
        "--no-persist",
    ]
    env = dict(os.environ)
    env["BASKETBALL_INFERENCE_CONF"] = str(baseline["inference_confidence"])
    env["BASKETBALL_BALL_CONF"] = str(baseline["ball_confidence"])
    env["BASKETBALL_BALL_REACQUIRE"] = "0"
    # P1.1 needs court_xy continuity as a diagnostic, while analytics remains off.
    env["BASKETBALL_COURT_CALIBRATION"] = "1"

    print(f"\n=== {video['label']} {video['path']} {video['start_sec']}–{video['end_sec']}s ===")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_dir = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        match = RUN_DIR_RE.match(line.strip())
        if match:
            run_dir = Path(match.group(1)).resolve()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Baseline run {video['label']} failed with exit code {return_code}")
    if run_dir is None:
        raise RuntimeError(f"Baseline run {video['label']} did not report [RUN_DIR]")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen CourtFlux P1.1 V1–V5 baseline")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default="mps")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    baseline = contract["canonical_baseline"]
    run_entries = []
    reports = []
    for video in contract["videos"]:
        run_dir = _run_one(video, baseline, args.device)
        report = evaluate_run(run_dir)
        reports.append(report)
        run_entries.append(
            {
                "label": video["label"],
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "video": video,
            }
        )

    payload = {
        "created_at": datetime.now().isoformat(),
        "contract": str(args.contract),
        "canonical_baseline": baseline,
        "runs": run_entries,
        "reports": reports,
        "aggregate": aggregate_reports(reports),
    }
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "evaluation" / "p1_tracking_reports" / f"baseline_v1_v5_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[P1.1 REPORT] {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
