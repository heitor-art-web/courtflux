from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_ROOT = PROJECT_ROOT / "evaluation" / "runs"


def _run_dirs() -> set[Path]:
    if not RUNS_ROOT.exists():
        return set()
    return {path.resolve() for path in RUNS_ROOT.iterdir() if path.is_dir()}


def _latest_run_dir(before: set[Path]) -> Path:
    candidates = _run_dirs() - before
    if not candidates:
        candidates = _run_dirs()
    if not candidates:
        raise RuntimeError("Analyzer completed without creating an evaluation run directory")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _build_probe_summary(run_dir: Path, backend: str) -> Path:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing analyzer summary: {summary_path}")

    summary = _read_json(summary_path)
    players = dict(summary.get("players") or {})
    identity = dict(players.get("identity_diagnostics") or {})
    debug_path = run_dir / "identity_debug" / "identity_debug_summary.json"
    debug = _read_json(debug_path) if debug_path.exists() else {}

    attempts_all = int(identity.get("jersey_attempts_all_inferred_frames") or 0)
    successes_all = int(identity.get("jersey_successes_all_inferred_frames") or 0)
    debug_attempts = int(debug.get("attempts") or 0)
    debug_accepted = int(debug.get("accepted") or 0)

    payload = {
        "version": "courtflux_jersey_recon_probe_v1",
        "run_id": summary.get("run_id") or run_dir.name,
        "video_source": summary.get("video_source"),
        "start_time_sec": summary.get("start_time_sec"),
        "end_time_sec": summary.get("end_time_sec"),
        "backend": backend,
        "jersey_interval_frames": summary.get("jersey_interval_frames"),
        "identified_tracks": identity.get("identified_tracks"),
        "jersey_identification_rate_metric_frames": players.get("jersey_identification_rate"),
        "attempts_all_inferred_frames": attempts_all,
        "successes_all_inferred_frames": successes_all,
        "acceptance_rate_all_inferred_attempts": _rate(successes_all, attempts_all),
        "debug_attempts": debug_attempts,
        "debug_accepted": debug_accepted,
        "debug_acceptance_rate": _rate(debug_accepted, debug_attempts),
        "high_quality_abstentions": debug.get("high_quality_abstentions"),
        "rejection_reasons": debug.get("rejection_reasons") or identity.get("rejection_reasons") or {},
        "candidate_counts": debug.get("candidate_counts") or identity.get("roster_candidate_counts") or {},
        "raw_prediction_counts": identity.get("raw_prediction_counts") or {},
        "checkpoint": identity.get("checkpoint") or {},
        "contact_sheets": debug.get("contact_sheets"),
        "analysis_note": (
            "These are automatic diagnostic/yield metrics, not accuracy. "
            "Upload the generated bundle for visual and failure-mode analysis."
        ),
    }
    output = run_dir / "jersey_recon_probe_summary.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _build_bundle(run_dir: Path) -> Path:
    bundle = run_dir / "jersey_recon_bundle.zip"
    include_files = [
        run_dir / "summary.json",
        run_dir / "run_config.json",
        run_dir / "match_config.json",
        run_dir / "frames.jsonl",
        run_dir / "jersey_recon_probe_summary.json",
        run_dir / "identity_debug" / "identity_debug_summary.json",
        run_dir / "identity_debug" / "identity_debug.jsonl",
    ]

    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in include_files:
            if path.exists() and path.is_file():
                archive.write(path, arcname=str(path.relative_to(run_dir)))

        contact_root = run_dir / "identity_debug" / "contact_sheets"
        if contact_root.exists():
            for path in sorted(contact_root.glob("*.jpg")):
                archive.write(path, arcname=str(path.relative_to(run_dir)))

    return bundle


def run_probe(args: argparse.Namespace) -> None:
    os.environ["BASKETBALL_JERSEY_BACKEND"] = args.jersey_backend
    os.environ["BASKETBALL_IDENTITY_DEBUG"] = "1"
    os.environ["BASKETBALL_SMART_BALL_MODE"] = "off"
    os.environ["BASKETBALL_SMART_BALL_PIPELINE"] = "0"

    from services.analytics import analyzer

    before = _run_dirs()
    analyzer.main(
        video_source=args.video,
        start_time=args.start,
        end_time=args.end,
        model_path=args.model,
        detector_conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        match_config_path=args.match_config,
        enable_jersey=True,
        jersey_interval_frames=args.jersey_interval,
        diagnostics_batch_size=args.diagnostics_batch_size,
        persist_results=False,
    )

    run_dir = _latest_run_dir(before)
    probe_summary_path = _build_probe_summary(run_dir, args.jersey_backend)
    bundle_path = _build_bundle(run_dir)
    payload = _read_json(probe_summary_path)

    print("\n[JERSEY_RECON_PROBE]", flush=True)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(f"[JERSEY_RECON_RUN_DIR] {run_dir}", flush=True)
    print(f"[JERSEY_RECON_SUMMARY] {probe_summary_path}", flush=True)
    print(f"[JERSEY_RECON_BUNDLE] {bundle_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CourtFlux Jersey recognition on a real video and package automatic "
            "diagnostics/contact sheets for offline analysis."
        )
    )
    parser.add_argument("--video", required=True, help="YouTube URL or local video path")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=30.0)
    parser.add_argument("--model", default="weights/model_v1.1.0.pt")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument(
        "--tracker",
        choices=("botsort.yaml", "bytetrack.yaml"),
        default="botsort.yaml",
    )
    parser.add_argument("--match-config", default="data/match_config.json")
    parser.add_argument("--jersey-backend", default="parseq_soccernet")
    parser.add_argument("--jersey-interval", type=int, default=24)
    parser.add_argument("--diagnostics-batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be greater than --start")
    if args.jersey_interval <= 0:
        parser.error("--jersey-interval must be greater than zero")
    return args


if __name__ == "__main__":
    run_probe(parse_args())
