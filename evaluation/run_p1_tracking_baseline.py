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


def _resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


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

    match_config = _resolve_repo_path(video.get("match_config"))
    if match_config is not None:
        if not match_config.exists():
            raise FileNotFoundError(
                f"P1.1 match config for {video['label']} does not exist: {match_config}"
            )
        command.extend(["--match-config", str(match_config)])

    env = dict(os.environ)
    env["BASKETBALL_INFERENCE_CONF"] = str(baseline["inference_confidence"])
    env["BASKETBALL_BALL_CONF"] = str(baseline["ball_confidence"])
    env["BASKETBALL_BALL_REACQUIRE"] = "0"
    # P1.1 needs court projection diagnostics while tactical analytics remains disabled.
    env["BASKETBALL_COURT_CALIBRATION"] = "1"

    print(
        f"\n=== {video['label']} {video['path']} {video['start_sec']}–{video['end_sec']}s "
        f"team_context={video.get('team_context', 'unspecified')} ==="
    )
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

    contract_path = args.contract if args.contract.is_absolute() else (ROOT / args.contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    baseline = contract["canonical_baseline"]
    eligibility_proxy = contract.get("analytics_eligibility_proxy", {}) or {}

    run_entries = []
    reports = []
    for video in contract["videos"]:
        run_dir = _run_one(video, baseline, args.device)
        benchmark_context = {
            "label": video["label"],
            "video": video["path"],
            "team_context": video.get("team_context", "unspecified"),
            "match_config": video.get("match_config"),
            "team_metrics_in_quality_gate": bool(video.get("team_metrics_in_quality_gate", False)),
            "analytics_eligibility_proxy": eligibility_proxy,
        }
        report = evaluate_run(run_dir, benchmark_context=benchmark_context)
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
        "contract": str(contract_path),
        "contract_schema_version": contract.get("schema_version"),
        "canonical_baseline": baseline,
        "analytics_eligibility_proxy": eligibility_proxy,
        "runs": run_entries,
        "reports": reports,
        "aggregate": aggregate_reports(reports),
    }
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT / "evaluation" / "p1_tracking_reports" / f"baseline_v2_v1_v5_{stamp}.json"
    elif not output.is_absolute():
        output = (ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[P1.1 REPORT] {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
