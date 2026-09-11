from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Settings, settings
from evaluation.player_tracking_quality import aggregate_reports, evaluate_run, load_frames
from services.vision.player_state_v1 import PlayerStateEstimatorV1


DEFAULT_CONTRACT = ROOT / "evaluation" / "p1_tracking_baseline_contract.json"
RUN_DIR_RE = re.compile(r"^\[RUN_DIR\]\s+(.+?)\s*$")
P1_CLASSES = ("basketball", "player", "referee", "rim", "backboard")
REQUIRED_PROVENANCE_KEYS = {
    "detector_model",
    "detector_version",
    "detector_baseline_status",
    "upstream_track_id",
    "upstream_class",
    "player_referee_separation",
    "tracking_association_owned_upstream",
    "identity_resolution",
    "uncertainty_policy",
}


def _resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _canonical_class(name: Any) -> str | None:
    value = str(name or "").strip().lower().replace("_", " ").replace("-", " ")
    if "basketball" in value or value in {"ball", "sports ball"}:
        return "basketball"
    if "referee" in value or value == "ref":
        return "referee"
    if "backboard" in value:
        return "backboard"
    if value == "rim" or "basket rim" in value:
        return "rim"
    if value == "player" or "basketball player" in value or value == "person":
        return "player"
    return None


def _raw_class_counts(frame: dict[str, Any]) -> Counter[str]:
    diagnostics = frame.get("diagnostics") or {}
    detector = diagnostics.get("detector") if isinstance(diagnostics, dict) else None
    raw = detector.get("raw_class_counts") if isinstance(detector, dict) else None
    counts: Counter[str] = Counter()
    if not isinstance(raw, dict):
        return counts
    for name, value in raw.items():
        canonical = _canonical_class(name)
        if canonical is None:
            continue
        try:
            counts[canonical] += max(0, int(value))
        except (TypeError, ValueError):
            continue
    return counts


def _tracked_people(frame: dict[str, Any]) -> list[dict[str, Any]]:
    rows = frame.get("players")
    if rows is None:
        rows = frame.get("team_predictions", [])
    output: list[dict[str, Any]] = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        row["class_name"] = "player"
        row.setdefault("source", "botsort_tracked_player")
        if bool(row.get("court_xy_valid")) and isinstance(row.get("court_xy"), (list, tuple)):
            row["court_position_m"] = list(row["court_xy"][:2])
        else:
            row.pop("court_position_m", None)
        output.append(row)
    return output


def _run_one(video: dict[str, Any], runtime: dict[str, Any], device: str) -> Path:
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
        str(runtime["model"]),
        "--conf",
        str(runtime["player_confidence"]),
        "--imgsz",
        str(runtime["imgsz"]),
        "--device",
        device,
        "--tracker",
        str(runtime["tracker"]),
        "--detection-only",
        "--no-jersey",
        "--no-persist",
    ]
    match_config = _resolve_repo_path(video.get("match_config"))
    if match_config is not None:
        if not match_config.exists():
            raise FileNotFoundError(
                f"P1 match config for {video['label']} does not exist: {match_config}"
            )
        command.extend(["--match-config", str(match_config)])

    env = dict(os.environ)
    env["BASKETBALL_INFERENCE_CONF"] = str(runtime["inference_confidence"])
    env["BASKETBALL_BALL_CONF"] = str(runtime["ball_confidence"])
    env["BASKETBALL_BALL_REACQUIRE"] = "0"
    env["BASKETBALL_COURT_CALIBRATION"] = "1"

    print(
        f"\n=== {video['label']} PlayerState | {video['path']} "
        f"{video['start_sec']}–{video['end_sec']}s | model={runtime['model']} "
        f"tracker={runtime['tracker']} player_conf={runtime['player_confidence']} ===",
        flush=True,
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
    run_dir: Path | None = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        match = RUN_DIR_RE.match(line.strip())
        if match:
            run_dir = Path(match.group(1)).resolve()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"PlayerState source run {video['label']} failed: {return_code}")
    if run_dir is None:
        raise RuntimeError(f"PlayerState source run {video['label']} did not report [RUN_DIR]")
    return run_dir


def _evaluate_player_states(
    *,
    label: str,
    run_dir: Path,
    config: Settings,
    output_dir: Path,
) -> dict[str, Any]:
    frames_path = run_dir / "frames.jsonl"
    frames = load_frames(frames_path)
    estimator = PlayerStateEstimatorV1(config=config)

    raw_classes: Counter[str] = Counter()
    diagnostics_total: Counter[str] = Counter()
    states_total = 0
    gameplay_states_total = 0
    gameplay_frames = 0
    frames_with_states = 0
    unique_track_ids: set[int] = set()
    court_position_states = 0
    speed_states = 0
    nonzero_velocity_states = 0
    provenance_complete = 0
    provenance_checked = 0
    confidence_sum = 0.0
    uncertainty_sum = 0.0

    states_path = output_dir / f"{label.lower()}_player_states.jsonl"
    with states_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            frame_id = int(frame.get("frame_id", 0) or 0)
            timestamp_raw = frame.get("timestamp_sec")
            timestamp_sec = None if timestamp_raw is None else float(timestamp_raw)
            gameplay = bool(frame.get("valid_gameplay", False))
            if gameplay:
                gameplay_frames += 1

            raw_classes.update(_raw_class_counts(frame))
            people = _tracked_people(frame)
            states = estimator.update(
                people,
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
            )
            diagnostics_total.update(estimator.last_diagnostics)
            states_total += len(states)
            if gameplay:
                gameplay_states_total += len(states)
            if states:
                frames_with_states += 1

            for state in states:
                payload = state.as_dict()
                payload["video_label"] = label
                payload["valid_gameplay"] = gameplay
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

                unique_track_ids.add(state.track_id)
                confidence_sum += state.confidence
                uncertainty_sum += state.uncertainty
                if state.court_position_m is not None:
                    court_position_states += 1
                if state.speed_mps is not None:
                    speed_states += 1
                if any(abs(value) > 1e-9 for value in state.velocity_px_per_frame):
                    nonzero_velocity_states += 1
                provenance_checked += 1
                if REQUIRED_PROVENANCE_KEYS.issubset(state.provenance):
                    provenance_complete += 1

    class_counts = {name: int(raw_classes.get(name, 0)) for name in P1_CLASSES}
    class_counts["total"] = sum(class_counts.values())
    print(
        f"[{label} PlayerState] detections["
        + " ".join(f"{name}={class_counts[name]}" for name in P1_CLASSES)
        + f" total={class_counts['total']}] state[states={states_total} "
        f"unique_tracks={len(unique_track_ids)} "
        f"untracked_rejected={diagnostics_total.get('player_missing_track_id', 0)}]",
        flush=True,
    )

    return {
        "label": label,
        "source_run_dir": str(run_dir),
        "source_frames": len(frames),
        "gameplay_frames": gameplay_frames,
        "player_states_file": str(states_path),
        "player_states_total": states_total,
        "player_states_per_source_frame": states_total / len(frames) if frames else 0.0,
        "player_states_per_gameplay_frame": (
            gameplay_states_total / gameplay_frames if gameplay_frames else 0.0
        ),
        "frames_with_player_states": frames_with_states,
        "unique_player_state_track_ids": len(unique_track_ids),
        "nonzero_velocity_states": nonzero_velocity_states,
        "court_position_states": court_position_states,
        "speed_states": speed_states,
        "mean_confidence": confidence_sum / states_total if states_total else None,
        "mean_uncertainty": uncertainty_sum / states_total if states_total else None,
        "provenance": {
            "required_keys": sorted(REQUIRED_PROVENANCE_KEYS),
            "frames_checked": provenance_checked,
            "complete": provenance_complete,
            "completeness_rate": (
                provenance_complete / provenance_checked if provenance_checked else 0.0
            ),
        },
        "detector_class_diagnostics": {
            "counts_total": class_counts,
            "semantics": (
                "Raw detector-class counts from analyzer diagnostics when available; "
                "absence means the source run did not expose that diagnostic, not zero GT objects."
            ),
        },
        "estimator_diagnostics_accumulated": dict(diagnostics_total),
        "separation_guard": {
            "raw_referee_detections": int(raw_classes.get("referee", 0)),
            "referee_player_states": 0,
            "referees_are_player_states": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run PlayerState v1 real-video validation on the frozen P1 tracking V1–V5 suite. "
            "Only the detector model is promoted to v1.1.0; BoT-SORT and tracking thresholds remain frozen."
        )
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--model", default=settings.model)
    parser.add_argument("--device", choices=["mps", "cuda", "cpu"], default=settings.device)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    contract_path = args.contract if args.contract.is_absolute() else (ROOT / args.contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    historical = dict(contract["canonical_baseline"])
    runtime = dict(historical)
    runtime["model"] = str(args.model)

    # Deliberately freeze every non-model tracking variable to the accepted baseline.
    for key in ("tracker", "imgsz", "player_confidence", "inference_confidence", "ball_confidence"):
        if runtime.get(key) != historical.get(key):
            raise AssertionError(f"Unexpected tracking baseline mutation for {key}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output.resolve()
        if args.output and args.output.is_absolute()
        else (ROOT / args.output).resolve()
        if args.output
        else ROOT / "evaluation" / "player_state_v1_real" / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    state_config = Settings(
        model=str(args.model),
        detector_version=settings.detector_version,
        detector_baseline_status=settings.detector_baseline_status,
        tracker=str(runtime["tracker"]),
        device=args.device,
        imgsz=int(runtime["imgsz"]),
    )

    run_entries: list[dict[str, Any]] = []
    tracking_reports: list[dict[str, Any]] = []
    state_reports: list[dict[str, Any]] = []
    eligibility_proxy = contract.get("analytics_eligibility_proxy", {}) or {}
    time_normalization = contract.get("time_normalization", {}) or {}

    for video in contract["videos"]:
        run_dir = _run_one(video, runtime, args.device)
        benchmark_context = {
            "label": video["label"],
            "video": video["path"],
            "team_context": video.get("team_context", "unspecified"),
            "match_config": video.get("match_config"),
            "team_metrics_in_quality_gate": bool(video.get("team_metrics_in_quality_gate", False)),
            "analytics_eligibility_proxy": eligibility_proxy,
            "time_normalization": time_normalization,
        }
        tracking_report = evaluate_run(run_dir, benchmark_context=benchmark_context)
        state_report = _evaluate_player_states(
            label=str(video["label"]),
            run_dir=run_dir,
            config=state_config,
            output_dir=output_dir,
        )
        tracking_reports.append(tracking_report)
        state_reports.append(state_report)
        run_entries.append(
            {
                "label": video["label"],
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "video": video,
            }
        )

    total_states = sum(int(row["player_states_total"]) for row in state_reports)
    total_provenance_checked = sum(int(row["provenance"]["frames_checked"]) for row in state_reports)
    total_provenance_complete = sum(int(row["provenance"]["complete"]) for row in state_reports)
    total_referees = sum(
        int(row["separation_guard"]["raw_referee_detections"]) for row in state_reports
    )
    total_untracked_rejected = sum(
        int(row["estimator_diagnostics_accumulated"].get("player_missing_track_id", 0))
        for row in state_reports
    )

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "P1 PlayerState v1 real-video validation",
        "contract": str(contract_path),
        "contract_schema_version": contract.get("schema_version"),
        "detector_promotion": {
            "historical_tracking_detector": historical.get("model"),
            "current_detector": str(args.model),
            "detector_version": settings.detector_version,
            "baseline_status": settings.detector_baseline_status,
            "only_detector_model_changed": True,
        },
        "frozen_tracking_runtime": {
            key: runtime.get(key)
            for key in (
                "tracker",
                "imgsz",
                "player_confidence",
                "inference_confidence",
                "ball_confidence",
                "detection_only",
                "jersey_enabled",
            )
        },
        "historical_baseline_findings": contract.get("baseline_findings", {}),
        "runs": run_entries,
        "player_state_reports": state_reports,
        "tracking_reports": tracking_reports,
        "tracking_aggregate": aggregate_reports(tracking_reports),
        "player_state_aggregate": {
            "videos": len(state_reports),
            "player_states_total": total_states,
            "untracked_player_observations_rejected": total_untracked_rejected,
            "raw_referee_detections": total_referees,
            "referee_player_states": 0,
            "provenance_completeness_rate": (
                total_provenance_complete / total_provenance_checked
                if total_provenance_checked
                else 0.0
            ),
            "mean_player_states_per_gameplay_frame_macro": (
                sum(float(row["player_states_per_gameplay_frame"]) for row in state_reports)
                / len(state_reports)
                if state_reports
                else 0.0
            ),
        },
        "interpretation_guard": {
            "true_player_recall_measured": False,
            "true_id_switches_measured": False,
            "true_fragmentation_by_physical_player_measured": False,
            "reason": "These require targeted player-identity ground truth per the frozen tracking contract.",
        },
        "scope_guard": {
            "team_identity_resolved": False,
            "jersey_identity_resolved": False,
            "possession": False,
            "pressure": False,
            "pass": False,
            "shot": False,
            "rebound": False,
            "p2_event_semantics": False,
        },
        "notes": [
            "V1–V5 and BoT-SORT tracking settings are inherited from the frozen P1 tracking baseline.",
            "Only the detector checkpoint is promoted from v1.0.0 to the current v1.1.0 baseline.",
            "PlayerState rejects untracked sentinel IDs such as -1 instead of inventing persistent identities.",
            "Referees remain separate detector evidence and never become PlayerState entries.",
            "Tracking quality formulas are reused from evaluation/player_tracking_quality.py for direct historical comparability.",
        ],
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nPlayerState v1 real-video suite complete")
    print(f"videos={len(state_reports)} states={total_states} untracked_rejected={total_untracked_rejected}")
    print(f"raw_referees={total_referees} referee_player_states=0")
    print(
        "provenance_completeness="
        f"{summary['player_state_aggregate']['provenance_completeness_rate']:.3f}"
    )
    print(f"[PLAYER_STATE_SUMMARY] {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
