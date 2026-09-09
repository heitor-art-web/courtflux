from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTS = [
    "evaluation/ball_candidate_backend_test.py",
    "evaluation/ball_candidate_pipeline_test.py",
    "evaluation/smart_candidate_filter_test.py",
    "evaluation/smart_ball_backend_integration_test.py",
    "evaluation/ball_hypothesis_tracker_test.py",
    "evaluation/ball_context_features_test.py",
    "evaluation/court_keypoint_map_test.py",
    "evaluation/system_quality_gate_test.py",
    "evaluation/p0_quality_gate_test.py",
]

FROZEN_PAIRS = [
    ("20260908_130530_855844", "20260907_121516_009508"),
    ("20260908_130718_769477", "20260907_121948_601883"),
    ("20260908_131032_611540", "20260907_122315_943004"),
    ("20260908_131204_724212", "20260907_122515_042534"),
    ("20260908_131609_006117", "20260907_122807_977379"),
]


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_inputs(run_root: Path, gt_root: Path) -> list[str]:
    missing: list[str] = []
    for run_id, gt_id in FROZEN_PAIRS:
        frames = run_root / run_id / "frames.jsonl"
        truth = gt_root / gt_id / "annotations.jsonl"
        if not frames.exists():
            missing.append(str(frames))
        if not truth.exists():
            missing.append(str(truth))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic P0 tests and the frozen V1-V5 system quality gate."
    )
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--output", default="evaluation/p0_quality_gate_smart_v1")
    args = parser.parse_args()

    if args.tests_only and args.gate_only:
        raise SystemExit("Choose at most one of --tests-only or --gate-only")

    if not args.gate_only:
        for test in TESTS:
            run_command([sys.executable, test])
        print("\n[P0 TESTS] PASS", flush=True)

    if args.tests_only:
        return

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    missing = validate_inputs(run_root, gt_root)
    if missing:
        print("\n[P0 GATE] SKIPPED: frozen run/GT files are missing:", flush=True)
        for path in missing:
            print(f"  - {path}", flush=True)
        if args.gate_only:
            raise SystemExit(2)
        return

    output = (ROOT / args.output).resolve()
    command = [
        sys.executable,
        "evaluation/p0_quality_gate.py",
        "--run-root",
        str(run_root),
        "--gt-root",
        str(gt_root),
        "--gt-radius-px",
        "40",
        "--catastrophic-radius-px",
        "160",
        "--output",
        str(output),
    ]
    for run_id, gt_id in FROZEN_PAIRS:
        command.extend(["--pair", f"{run_id}={gt_id}"])

    run_command(command)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = str((summary.get("gate") or {}).get("status") or "UNKNOWN")
    print(f"\n[P0 VALIDATION] {status} {summary_path}", flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
