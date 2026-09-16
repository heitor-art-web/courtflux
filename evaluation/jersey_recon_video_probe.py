from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                rows.append(row)
    return rows


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
    debug_root = run_dir / "identity_debug"
    debug_path = debug_root / "identity_debug_summary.json"
    debug = _read_json(debug_path) if debug_path.exists() else {}
    debug_rows = _read_jsonl(debug_root / "identity_debug.jsonl")

    attempts_all = int(identity.get("jersey_attempts_all_inferred_frames") or 0)
    successes_all = int(identity.get("jersey_successes_all_inferred_frames") or 0)
    debug_attempts = int(debug.get("attempts") or len(debug_rows))
    debug_accepted = int(debug.get("accepted") or 0)
    consensus_numeric_rows = [row for row in debug_rows if row.get("raw_canonical") is not None]
    consensus_numeric_tracks = {
        int(row.get("track_id"))
        for row in consensus_numeric_rows
        if isinstance(row.get("track_id"), int) and int(row.get("track_id")) >= 0
    }

    crop_total = 0
    crop_numeric = 0
    attempts_with_agreement = 0
    attempts_with_conflict = 0
    attempts_without_regions = 0
    support_counts: Counter[int] = Counter()
    selected_image_counts: Counter[int] = Counter()
    roi_stats: dict[str, Counter[str]] = {}
    region_scores: list[float] = []

    for row in debug_rows:
        per_crop = [
            item
            for item in list(row.get("per_crop_predictions") or [])
            if isinstance(item, dict)
        ]
        consensus = dict(row.get("multiframe_consensus") or {})
        selected_images = consensus.get("selected_images")
        if isinstance(selected_images, int):
            selected_image_counts[int(selected_images)] += 1
        if not per_crop:
            attempts_without_regions += 1
        else:
            crop_total += len(per_crop)
            crop_numeric += sum(
                1 for item in per_crop if item.get("raw_canonical") is not None
            )
            numbers = [
                str(item.get("raw_canonical"))
                for item in per_crop
                if item.get("raw_canonical") is not None
            ]
            unique = set(numbers)
            if numbers and len(unique) == 1:
                attempts_with_agreement += 1
            elif len(unique) > 1:
                attempts_with_conflict += 1

            final_number = row.get("raw_canonical")
            for item in per_crop:
                roi_name = str(item.get("roi_name") or "unknown")
                stats = roi_stats.setdefault(roi_name, Counter())
                stats["outputs"] += 1
                if item.get("raw_canonical") is not None:
                    stats["numeric_outputs"] += 1
                if final_number is not None and str(item.get("raw_canonical")) == str(final_number):
                    stats["votes_for_final_consensus"] += 1
                score = item.get("roi_prior")
                if isinstance(score, (int, float)):
                    region_scores.append(float(score))

        support = row.get("candidate_count")
        if isinstance(support, int):
            support_counts[int(support)] += 1

    roi_payload: dict[str, Any] = {}
    for name, stats in sorted(roi_stats.items()):
        outputs = int(stats.get("outputs", 0))
        numeric = int(stats.get("numeric_outputs", 0))
        votes = int(stats.get("votes_for_final_consensus", 0))
        roi_payload[name] = {
            "outputs": outputs,
            "numeric_outputs": numeric,
            "numeric_yield": _rate(numeric, outputs),
            "votes_for_final_consensus": votes,
            "consensus_vote_rate": _rate(votes, numeric),
        }

    payload = {
        "version": "courtflux_jersey_recon_probe_v4_number_region",
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
        "consensus_numeric_outputs": len(consensus_numeric_rows),
        "consensus_numeric_yield": _rate(len(consensus_numeric_rows), debug_attempts),
        "tracks_with_consensus_numeric_output": len(consensus_numeric_tracks),
        "region_proposal_outputs": crop_total,
        "region_numeric_outputs": crop_numeric,
        "region_numeric_yield": _rate(crop_numeric, crop_total),
        "attempts_without_region_proposals": attempts_without_regions,
        "region_proposals_per_attempt": round(crop_total / max(debug_attempts, 1), 4),
        "mean_region_score": (
            round(sum(region_scores) / len(region_scores), 6) if region_scores else 0.0
        ),
        "attempts_all_region_outputs_agree": attempts_with_agreement,
        "attempts_region_outputs_conflict": attempts_with_conflict,
        "support_histogram": {
            str(key): value for key, value in sorted(support_counts.items())
        },
        "selected_images_histogram": {
            str(key): value for key, value in sorted(selected_image_counts.items())
        },
        "region_rank_stats": roi_payload,
        "high_quality_abstentions": debug.get("high_quality_abstentions"),
        "rejection_reasons": debug.get("rejection_reasons")
        or identity.get("rejection_reasons")
        or {},
        "candidate_counts": debug.get("candidate_counts")
        or identity.get("roster_candidate_counts")
        or {},
        "raw_prediction_counts": identity.get("raw_prediction_counts") or {},
        "checkpoint": identity.get("checkpoint") or {},
        "contact_sheets": debug.get("contact_sheets"),
        "analysis_note": (
            "v4 is a roster-blind, OCR-blind visual number-region proposal experiment. "
            "Region proposal yield measures how selectively the visual scorer gates PARSeq; "
            "consensus yield still measures stability, not ground-truth accuracy."
        ),
    }
    output = run_dir / "jersey_recon_probe_summary.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
    os.environ["BASKETBALL_JERSEY_RESEARCH_MULTIFRAME"] = "1"
    os.environ["BASKETBALL_JERSEY_RESEARCH_ROI"] = "1"
    os.environ["BASKETBALL_JERSEY_RESEARCH_TOP_K"] = str(args.top_k)
    os.environ["BASKETBALL_JERSEY_RESEARCH_MIN_SUPPORT"] = str(args.min_support)
    os.environ["BASKETBALL_JERSEY_RESEARCH_MIN_SHARE"] = str(args.min_share)

    from services.analytics import analyzer
    from services.vision.research_jersey_region_analyzer import (
        ResearchNumberRegionPARSeqAnalyzer,
    )

    def _research_build_jersey_visual(
        *, profiling_path: str, device: str, project_root: Path
    ) -> Any:
        del project_root
        if args.jersey_backend != "parseq_soccernet":
            raise ValueError(
                "The v4 number-region probe currently supports only parseq_soccernet"
            )
        visual = ResearchNumberRegionPARSeqAnalyzer(
            profiling_path=profiling_path,
            device=device,
        )
        checkpoint = dict(getattr(visual, "checkpoint_diagnostics", {}) or {})
        checkpoint["requested_backend"] = args.jersey_backend
        checkpoint["backend_contract_verified"] = True
        visual.checkpoint_diagnostics = checkpoint
        print(
            "[JERSEY_RESEARCH_REGION] method=contrast_components_geometry_v1 "
            f"min_score={visual.REGION_MIN_SCORE:.2f} max_regions={visual.REGION_MAX_PROPOSALS}",
            flush=True,
        )
        return visual

    analyzer.build_jersey_visual = _research_build_jersey_visual

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
            "Run CourtFlux Jersey recognition with research number-region proposals "
            "and package diagnostics for offline analysis."
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
        "--tracker", choices=("botsort.yaml", "bytetrack.yaml"), default="botsort.yaml"
    )
    parser.add_argument("--match-config", default="data/match_config.json")
    parser.add_argument("--jersey-backend", default="parseq_soccernet")
    parser.add_argument("--jersey-interval", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-share", type=float, default=0.60)
    parser.add_argument("--diagnostics-batch-size", type=int, default=100)
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be greater than --start")
    if args.jersey_interval <= 0:
        parser.error("--jersey-interval must be greater than zero")
    if not 2 <= args.top_k <= 8:
        parser.error("--top-k must be between 2 and 8")
    if not 2 <= args.min_support <= args.top_k:
        parser.error("--min-support must be between 2 and --top-k")
    if not 0.50 <= args.min_share <= 0.95:
        parser.error("--min-share must be between 0.50 and 0.95")
    return args


if __name__ == "__main__":
    run_probe(parse_args())
