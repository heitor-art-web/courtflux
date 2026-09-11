from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Settings, settings
from evaluation.player_state_v1_real_eval import _evaluate_player_states, _run_one
from evaluation.player_tracking_quality import aggregate_reports, evaluate_run

DEFAULT_CONTRACT = ROOT / "evaluation" / "p1_tracking_baseline_contract.json"
DEFAULT_THRESHOLDS = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def _benchmark_context(contract: dict[str, Any], video: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": video["label"],
        "video": video["path"],
        "team_context": video.get("team_context", "unspecified"),
        "match_config": video.get("match_config"),
        "team_metrics_in_quality_gate": bool(video.get("team_metrics_in_quality_gate", False)),
        "analytics_eligibility_proxy": contract.get("analytics_eligibility_proxy", {}) or {},
        "time_normalization": contract.get("time_normalization", {}) or {},
    }


def _metrics(report: dict[str, Any], threshold: float, run_dir: Path) -> dict[str, Any]:
    frames = report.get("frames", {}) or {}
    eligible = frames.get("analytics_eligible", {}) or {}
    bridge = report.get("detector_tracker_bridge", {}) or {}
    tracking = report.get("tracking", {}) or {}
    time_norm = tracking.get("time_normalized", {}) or {}
    return {
        "player_confidence": float(threshold),
        "run_dir": str(run_dir),
        "analytics_eligible_players_per_frame": eligible.get("players_per_frame_mean"),
        "tracker_assignment_rate_pct": bridge.get("tracker_assignment_rate_pct"),
        "untracked_observation_rate_pct": bridge.get("untracked_observation_rate_pct"),
        "short_track_rate_pct": tracking.get("short_track_rate_pct"),
        "time_normalized_short_track_rate_pct": time_norm.get("short_track_rate_pct"),
        "same_id_interruptions_per_gameplay_minute": time_norm.get(
            "same_id_interruptions_per_gameplay_minute"
        ),
        "unique_track_ids_per_1000_tracked_observations": time_norm.get(
            "unique_track_ids_per_1000_tracked_observations"
        ),
        "unique_track_ids": tracking.get("unique_track_ids"),
        "tracked_observations": tracking.get("tracked_observations"),
        "untracked_observations": tracking.get("untracked_observations"),
    }


def choose_stress_threshold(
    rows: list[dict[str, Any]], *, max_players_proxy_drop_pct: float = 5.0
) -> dict[str, Any]:
    if not rows:
        raise ValueError("stress threshold sweep produced no rows")
    baseline = min(rows, key=lambda row: abs(float(row["player_confidence"]) - 0.10))
    baseline_players = float(baseline.get("analytics_eligible_players_per_frame") or 0.0)
    floor = baseline_players * (1.0 - max_players_proxy_drop_pct / 100.0)

    viable = [
        row
        for row in rows
        if float(row.get("analytics_eligible_players_per_frame") or 0.0) >= floor
    ]
    pool = viable or rows

    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        assignment = float(row.get("tracker_assignment_rate_pct") or 0.0)
        untracked = float(row.get("untracked_observation_rate_pct") or 100.0)
        short_time = float(row.get("time_normalized_short_track_rate_pct") or 100.0)
        churn = float(row.get("unique_track_ids_per_1000_tracked_observations") or 1e9)
        threshold = float(row["player_confidence"])
        return (assignment, -untracked, -short_time, -churn, -threshold)

    selected = max(pool, key=key)
    return {
        "selected_player_confidence": float(selected["player_confidence"]),
        "selected_metrics": selected,
        "reference_010_metrics": baseline,
        "analytics_eligible_players_per_frame_floor": floor,
        "max_players_proxy_drop_pct": float(max_players_proxy_drop_pct),
        "viable_thresholds": [float(row["player_confidence"]) for row in viable],
        "selection_semantics": (
            "Stress-video calibration only. Preserve at least 95% of the v1.1.0 @ 0.10 "
            "analytics-eligible players/frame proxy, then prioritize tracker assignment, "
            "untracked rate, time-normalized short tracks and track churn."
        ),
    }


def evaluate_full_gate(
    aggregate: dict[str, Any],
    historical: dict[str, Any],
    *,
    assignment_regression_pp: float = 0.5,
    untracked_regression_pp: float = 0.5,
    short_track_regression_pp: float = 2.0,
    players_proxy_drop_pct: float = 5.0,
) -> dict[str, Any]:
    current_players = float(aggregate.get("analytics_eligible_players_per_frame_macro_mean") or 0.0)
    current_assignment = float(aggregate.get("tracker_assignment_rate_macro_pct") or 0.0)
    current_untracked = float(aggregate.get("untracked_observation_rate_macro_pct") or 100.0)
    current_short = float(aggregate.get("short_track_rate_macro_pct") or 100.0)

    hist_players = float(historical.get("analytics_eligible_players_per_frame_macro_mean") or 0.0)
    hist_assignment = float(historical.get("tracker_assignment_rate_macro_pct") or 0.0)
    hist_untracked = float(historical.get("untracked_observation_rate_macro_pct") or 0.0)
    hist_short = float(historical.get("short_track_rate_frame_based_macro_pct") or 0.0)

    checks = {
        "players_proxy": current_players >= hist_players * (1.0 - players_proxy_drop_pct / 100.0),
        "tracker_assignment": current_assignment >= hist_assignment - assignment_regression_pp,
        "untracked_observations": current_untracked <= hist_untracked + untracked_regression_pp,
        "short_tracks": current_short <= hist_short + short_track_regression_pp,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "limits": {
            "assignment_regression_pp": assignment_regression_pp,
            "untracked_regression_pp": untracked_regression_pp,
            "short_track_regression_pp": short_track_regression_pp,
            "players_proxy_drop_pct": players_proxy_drop_pct,
        },
        "deltas": {
            "analytics_eligible_players_per_frame": current_players - hist_players,
            "tracker_assignment_rate_pp": current_assignment - hist_assignment,
            "untracked_observation_rate_pp": current_untracked - hist_untracked,
            "short_track_rate_pp": current_short - hist_short,
        },
        "historical": historical,
        "current": {
            "analytics_eligible_players_per_frame_macro_mean": current_players,
            "tracker_assignment_rate_macro_pct": current_assignment,
            "untracked_observation_rate_macro_pct": current_untracked,
            "short_track_rate_macro_pct": current_short,
        },
    }


def _parse_thresholds(value: str) -> list[float]:
    output: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        threshold = float(item)
        if not 0.0 < threshold <= 1.0:
            raise argparse.ArgumentTypeError("thresholds must be in (0, 1]")
        output.append(threshold)
    if not output:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    return sorted(set(output))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate model_v1.1.0 player confidence on the frozen BoT-SORT P1 tracking "
            "contract. V2 is the stress-video tuning split; the selected threshold is then "
            "validated on the complete V1–V5 suite."
        )
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model", default="weights/model_v1.1.0.pt")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=settings.device)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=list(DEFAULT_THRESHOLDS),
        help="Comma-separated player confidence thresholds.",
    )
    parser.add_argument("--stress-label", default="V2")
    parser.add_argument("--max-players-proxy-drop-pct", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    contract_path = args.contract if args.contract.is_absolute() else (ROOT / args.contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    historical_runtime = dict(contract["canonical_baseline"])
    historical_findings = dict(contract.get("baseline_findings", {}))
    videos = list(contract["videos"])
    stress_video = next((video for video in videos if video.get("label") == args.stress_label), None)
    if stress_video is None:
        raise ValueError(f"stress label not found in contract: {args.stress_label}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.resolve()
        if args.output and args.output.is_absolute()
        else (ROOT / args.output).resolve()
        if args.output
        else ROOT / "evaluation" / "player_model_v110_calibration" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stress_rows: list[dict[str, Any]] = []
    stress_runs: dict[float, Path] = {}
    for threshold in args.thresholds:
        runtime = dict(historical_runtime)
        runtime["model"] = str(args.model)
        runtime["player_confidence"] = float(threshold)
        print(f"\n[PLAYER_CALIBRATION] stress={args.stress_label} player_conf={threshold:.3f}")
        run_dir = _run_one(stress_video, runtime, args.device)
        report = evaluate_run(run_dir, benchmark_context=_benchmark_context(contract, stress_video))
        row = _metrics(report, threshold, run_dir)
        stress_rows.append(row)
        stress_runs[float(threshold)] = run_dir
        print(
            "[PLAYER_CALIBRATION_RESULT] "
            f"conf={threshold:.3f} players_proxy={row['analytics_eligible_players_per_frame']} "
            f"assignment={row['tracker_assignment_rate_pct']} "
            f"untracked={row['untracked_observation_rate_pct']} "
            f"short_time={row['time_normalized_short_track_rate_pct']} "
            f"churn={row['unique_track_ids_per_1000_tracked_observations']}"
        )

    selection = choose_stress_threshold(
        stress_rows, max_players_proxy_drop_pct=args.max_players_proxy_drop_pct
    )
    selected = float(selection["selected_player_confidence"])
    print(f"\n[PLAYER_CALIBRATION_SELECTED] player_conf={selected:.3f}")

    validation_reports: list[dict[str, Any]] = []
    state_reports: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []
    state_config = Settings(
        model=str(args.model),
        detector_version=settings.detector_version,
        detector_baseline_status=settings.detector_baseline_status,
        tracker=str(historical_runtime["tracker"]),
        device=args.device,
        imgsz=int(historical_runtime["imgsz"]),
    )

    for video in videos:
        label = str(video["label"])
        if label == args.stress_label and selected in stress_runs:
            run_dir = stress_runs[selected]
            print(f"\n[PLAYER_CALIBRATION] reusing {label} stress run for full validation")
        else:
            runtime = dict(historical_runtime)
            runtime["model"] = str(args.model)
            runtime["player_confidence"] = selected
            run_dir = _run_one(video, runtime, args.device)
        report = evaluate_run(run_dir, benchmark_context=_benchmark_context(contract, video))
        validation_reports.append(report)
        state_reports.append(
            _evaluate_player_states(
                label=label,
                run_dir=run_dir,
                config=state_config,
                output_dir=output_dir,
            )
        )
        run_entries.append({"label": label, "run_dir": str(run_dir), "video": video})

    validation_aggregate = aggregate_reports(validation_reports)
    gate = evaluate_full_gate(validation_aggregate, historical_findings)
    total_states = sum(int(row["player_states_total"]) for row in state_reports)
    total_rejected = sum(
        int(row["estimator_diagnostics_accumulated"].get("player_missing_track_id", 0))
        for row in state_reports
    )
    provenance_checked = sum(int(row["provenance"]["frames_checked"]) for row in state_reports)
    provenance_complete = sum(int(row["provenance"]["complete"]) for row in state_reports)

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "model_v1.1.0 player tracking threshold calibration for PlayerState v1",
        "evidence_status": "stress_video_calibration_plus_v1_v5_operational_validation_not_identity_gt",
        "contract": str(contract_path),
        "model": str(args.model),
        "frozen_runtime": {
            key: historical_runtime.get(key)
            for key in ("tracker", "imgsz", "inference_confidence", "ball_confidence")
        },
        "stress_video": args.stress_label,
        "thresholds": list(args.thresholds),
        "stress_sweep": stress_rows,
        "selection": selection,
        "full_validation_runs": run_entries,
        "full_validation_tracking_reports": validation_reports,
        "full_validation_aggregate": validation_aggregate,
        "full_validation_gate": gate,
        "player_state_reports": state_reports,
        "player_state_aggregate": {
            "videos": len(state_reports),
            "player_states_total": total_states,
            "untracked_player_observations_rejected": total_rejected,
            "provenance_completeness_rate": (
                provenance_complete / provenance_checked if provenance_checked else 0.0
            ),
        },
        "interpretation_guard": {
            "true_player_recall_measured": False,
            "true_id_switches_measured": False,
            "true_fragmentation_by_physical_player_measured": False,
            "selection_tuned_on_stress_video": True,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[PLAYER_CALIBRATION_SUMMARY] {summary_path.resolve()}")
    print(
        "[PLAYER_CALIBRATION_GATE] "
        f"pass={gate['pass']} checks={gate['checks']} deltas={gate['deltas']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
