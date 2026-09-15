from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class P2MetricsCollector:
    """Aggregate exploratory P2 metrics without changing runtime decisions."""

    def __init__(self) -> None:
        self.frames_seen = 0
        self.p2_processed_frames = 0
        self.metric_eligible_frames = 0
        self.engine_eligible_frames = 0
        self.possessed_frames = 0
        self.handler_memory_frames = 0
        self.possession_confidence_total = 0.0
        self.possession_confidence_samples = 0
        self.p1_not_mutated_frames = 0
        self.event_counts: Counter[str] = Counter()
        self.decision_counts: Counter[str] = Counter()
        self.shot_outcomes: Counter[str] = Counter()
        self.event_confidence_total: Counter[str] = Counter()
        self.event_confidence_samples: Counter[str] = Counter()
        self.control_state_counts: Counter[str] = Counter()
        self.reset_reasons: Counter[str] = Counter()
        self.events_with_actor = 0
        self.events_with_team = 0
        self.event_actor_counts: Counter[str] = Counter()
        self.event_team_counts: Counter[str] = Counter()
        self.total_events = 0
        self.fps = 30.0
        self.max_reset_count = 0
        self.event_rows: list[dict[str, Any]] = []
        self.state_rows: list[dict[str, Any]] = []

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / max(denominator, 1), 2)

    def observe(self, frame_state: Any) -> None:
        self.frames_seen += 1
        try:
            self.fps = max(1.0, float(getattr(frame_state, "fps", self.fps) or self.fps))
        except (TypeError, ValueError):
            pass

        metric_eligible = bool(getattr(frame_state, "valid_gameplay", False))
        if metric_eligible:
            self.metric_eligible_frames += 1

        diagnostics = dict(getattr(frame_state, "diagnostics", {}) or {})
        p2 = dict(diagnostics.get("p2", {}) or {})
        if not p2:
            return
        self.p2_processed_frames += 1

        if bool(p2.get("p1_state_is_not_mutated", False)):
            self.p1_not_mutated_frames += 1

        engine_eligible = bool(p2.get("engine_eligible", metric_eligible))
        if engine_eligible:
            self.engine_eligible_frames += 1

        possession = dict(p2.get("possession", {}) or {})
        current_track_id = possession.get("current_track_id")
        control_state = str(possession.get("control_state") or "unknown")
        if engine_eligible:
            self.control_state_counts[control_state] += 1

        if metric_eligible and engine_eligible and current_track_id is not None:
            self.handler_memory_frames += 1

        if metric_eligible and engine_eligible and control_state == "possessed":
            self.possessed_frames += 1
            try:
                confidence = float(possession.get("current_confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            self.possession_confidence_total += confidence
            self.possession_confidence_samples += 1

        reset = dict(p2.get("temporal_reset", {}) or {})
        try:
            reset_count = max(0, int(reset.get("count", 0) or 0))
        except (TypeError, ValueError):
            reset_count = 0
        if reset_count > self.max_reset_count:
            delta = reset_count - self.max_reset_count
            reason = str(reset.get("last_reason") or "unknown")
            self.reset_reasons[reason] += delta
            self.max_reset_count = reset_count

        shot_diag = dict(p2.get("shot", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        self.state_rows.append(
            {
                "frame_id": int(getattr(frame_state, "frame_id", 0)),
                "timestamp_sec": float(getattr(frame_state, "timestamp", 0.0) or 0.0),
                "metric_eligible": metric_eligible,
                "engine_eligible": engine_eligible,
                "control_state": control_state,
                "current_track_id": current_track_id,
                "current_team_id": possession.get("current_team_id"),
                "current_confidence": possession.get("current_confidence"),
                "lost_frames": possession.get("lost_frames"),
                "flight_active": bool(flight.get("active", False)),
                "flight_supported": bool(flight.get("flight_supported", False)),
                "flight_frames": flight.get("flight_frames"),
                "shot_active": bool(shot_diag.get("active", False)),
                "shot_awaiting_rearm": bool(shot_diag.get("awaiting_rearm", False)),
                "shot_rearm_clear_observed_frames": shot_diag.get("rearm_clear_observed_frames"),
                "reset_count": reset_count,
                "reset_reason": reset.get("last_reason"),
            }
        )

        for event in list(p2.get("events", []) or []):
            if not isinstance(event, dict):
                continue
            row = dict(event)
            event_type = str(row.get("event_type") or "unknown")
            decision = str(row.get("decision") or "unknown")
            self.event_counts[event_type] += 1
            self.decision_counts[decision] += 1
            self.total_events += 1

            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            self.event_confidence_total[event_type] += confidence
            self.event_confidence_samples[event_type] += 1

            if row.get("actor_track_id") is not None:
                self.events_with_actor += 1
                self.event_actor_counts[event_type] += 1
            if row.get("team_id") not in {None, "", "unknown", "UNKNOWN"}:
                self.events_with_team += 1
                self.event_team_counts[event_type] += 1

            if event_type == "shot_attempt":
                attributes = dict(row.get("attributes", {}) or {})
                self.shot_outcomes[str(attributes.get("outcome") or "unknown")] += 1

            self.event_rows.append(row)

    def _shot_retrigger_metrics(self) -> dict[str, Any]:
        shots = [
            row
            for row in self.event_rows
            if str(row.get("event_type") or "") == "shot_attempt"
        ]
        shots.sort(key=lambda row: int(row.get("frame_start", 0)))
        threshold_frames = max(1, int(round(self.fps * 0.30)))
        rapid_unknown_retriggers = 0
        min_gap_frames: int | None = None
        for previous, current in zip(shots, shots[1:]):
            previous_outcome = str(
                dict(previous.get("attributes", {}) or {}).get("outcome") or "unknown"
            )
            gap = int(current.get("frame_start", 0)) - int(previous.get("frame_end", 0))
            if min_gap_frames is None or gap < min_gap_frames:
                min_gap_frames = gap
            if previous_outcome == "unknown" and 0 <= gap <= threshold_frames:
                rapid_unknown_retriggers += 1
        return {
            "rapid_unknown_retrigger_window_frames": threshold_frames,
            "rapid_unknown_retrigger_count": rapid_unknown_retriggers,
            "minimum_inter_shot_gap_frames": min_gap_frames,
        }

    def summary(self) -> dict[str, Any]:
        avg_possession_confidence = (
            self.possession_confidence_total / self.possession_confidence_samples
            if self.possession_confidence_samples
            else None
        )
        event_confidence_mean = {
            event_type: round(
                self.event_confidence_total[event_type]
                / max(self.event_confidence_samples[event_type], 1),
                4,
            )
            for event_type in sorted(self.event_confidence_samples)
        }
        attribution_by_type = {
            event_type: {
                "total": int(count),
                "actor_count": int(self.event_actor_counts.get(event_type, 0)),
                "actor_rate": self._rate(
                    int(self.event_actor_counts.get(event_type, 0)),
                    int(count),
                ),
                "team_count": int(self.event_team_counts.get(event_type, 0)),
                "team_rate": self._rate(
                    int(self.event_team_counts.get(event_type, 0)),
                    int(count),
                ),
            }
            for event_type, count in sorted(self.event_counts.items())
        }
        shot_attempts = int(self.event_counts.get("shot_attempt", 0))
        unknown_shots = int(self.shot_outcomes.get("unknown", 0))

        passes_with_flight = 0
        turnovers_with_flight = 0
        direct_turnovers = 0
        for row in self.event_rows:
            event_type = str(row.get("event_type") or "")
            attrs = dict(row.get("attributes", {}) or {})
            flight_supported = bool(attrs.get("flight_supported", False))
            if event_type == "pass_candidate" and flight_supported:
                passes_with_flight += 1
            if event_type == "turnover_candidate":
                if flight_supported:
                    turnovers_with_flight += 1
                else:
                    direct_turnovers += 1

        control_rates = {
            state: self._rate(count, self.engine_eligible_frames)
            for state, count in sorted(self.control_state_counts.items())
        }

        return {
            "version": "p2_real_video_probe_v3",
            "frames_seen": self.frames_seen,
            "p2_processed_frames": self.p2_processed_frames,
            "metric_eligible_frames": self.metric_eligible_frames,
            "engine_eligible_frames": self.engine_eligible_frames,
            "control_state": {
                "counts": dict(sorted(self.control_state_counts.items())),
                "rates": control_rates,
            },
            "possession": {
                "possessed_frames": self.possessed_frames,
                "possessed_rate": self._rate(
                    self.possessed_frames,
                    self.engine_eligible_frames,
                ),
                "handler_memory_frames": self.handler_memory_frames,
                "handler_memory_rate": self._rate(
                    self.handler_memory_frames,
                    self.engine_eligible_frames,
                ),
                "avg_possession_confidence": (
                    None
                    if avg_possession_confidence is None
                    else round(avg_possession_confidence, 4)
                ),
                "changes": int(self.event_counts.get("possession_change", 0)),
                "acquisitions": int(self.event_counts.get("possession_acquired", 0)),
            },
            "events": {
                "total": self.total_events,
                "counts": dict(sorted(self.event_counts.items())),
                "decisions": dict(sorted(self.decision_counts.items())),
                "mean_confidence_by_type": event_confidence_mean,
                "actor_attribution_rate": self._rate(
                    self.events_with_actor,
                    self.total_events,
                ),
                "team_attribution_rate": self._rate(
                    self.events_with_team,
                    self.total_events,
                ),
                "attribution_by_event_type": attribution_by_type,
            },
            "shots": {
                "attempts": shot_attempts,
                "outcomes": dict(sorted(self.shot_outcomes.items())),
                "unknown_outcome_rate": self._rate(unknown_shots, shot_attempts),
                **self._shot_retrigger_metrics(),
            },
            "transitions": {
                "passes": int(self.event_counts.get("pass_candidate", 0)),
                "passes_with_flight_evidence": passes_with_flight,
                "turnovers": int(self.event_counts.get("turnover_candidate", 0)),
                "turnovers_with_flight_evidence": turnovers_with_flight,
                "direct_turnovers": direct_turnovers,
                "loose_ball_started": int(self.event_counts.get("loose_ball_started", 0)),
                "loose_ball_recovered": int(self.event_counts.get("loose_ball_recovered", 0)),
            },
            "passes": int(self.event_counts.get("pass_candidate", 0)),
            "turnovers": int(self.event_counts.get("turnover_candidate", 0)),
            "rebounds": int(self.event_counts.get("rebound_candidate", 0)),
            "resets": {
                "count": self.max_reset_count,
                "reasons": dict(sorted(self.reset_reasons.items())),
            },
            "p1_state_not_mutated_rate": self._rate(
                self.p1_not_mutated_frames,
                self.p2_processed_frames,
            ),
        }


def _install_probe(analyzer_module: Any, p2_pipeline_cls: Any) -> None:
    """Patch only this probe process; the normal Lab/runtime path stays untouched."""

    base_diagnostics = analyzer_module.RunDiagnostics

    class P2ProbeDiagnostics(base_diagnostics):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.p2_metrics = P2MetricsCollector()

        def log_frame(self, frame_state: Any) -> None:
            super().log_frame(frame_state)
            self.p2_metrics.observe(frame_state)

        def finish(self, extra=None, performance=None):
            p2_summary = self.p2_metrics.summary()
            merged_extra = dict(extra or {})
            merged_extra["p2_probe"] = p2_summary
            path = super().finish(extra=merged_extra, performance=performance)

            metrics_path = self.run_dir / "p2_metrics.json"
            metrics_path.write_text(
                json.dumps(p2_summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            events_path = self.run_dir / "p2_events.jsonl"
            events_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in self.p2_metrics.event_rows
                ),
                encoding="utf-8",
            )
            states_path = self.run_dir / "p2_states.jsonl"
            states_path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in self.p2_metrics.state_rows
                ),
                encoding="utf-8",
            )
            print(f"[P2_METRICS] {json.dumps(p2_summary, separators=(',', ':'))}", flush=True)
            print(f"[P2_METRICS_FILE] {metrics_path.resolve()}", flush=True)
            print(f"[P2_EVENTS_FILE] {events_path.resolve()}", flush=True)
            print(f"[P2_STATES_FILE] {states_path.resolve()}", flush=True)
            return path

    analyzer_module.RunDiagnostics = P2ProbeDiagnostics
    analyzer_module.Pipeline = p2_pipeline_cls


def run_probe(args: argparse.Namespace) -> None:
    # Freeze the PR #43 / rebuilt-P2 baseline before runtime construction. Smart Ball is
    # explicitly OFF so real-video comparisons do not silently inherit its legacy default.
    os.environ["BASKETBALL_P2_ENABLED"] = "1"
    os.environ["BASKETBALL_SMART_BALL_MODE"] = "off"
    os.environ["BASKETBALL_SMART_BALL_PIPELINE"] = "0"

    # Heavy runtime imports are deliberately delayed so P2 metric aggregation remains
    # unit-testable in lightweight CI without OpenCV/Ultralytics/weights.
    from services.analytics import analyzer
    from services.p2.pipeline import P2Pipeline

    _install_probe(analyzer, P2Pipeline)

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
        enable_jersey=not args.no_jersey,
        jersey_interval_frames=args.jersey_interval,
        diagnostics_batch_size=args.diagnostics_batch_size,
        persist_results=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated P2 real-video probe using the canonical Lab analyzer.",
    )
    parser.add_argument(
        "--video",
        default="https://www.youtube.com/watch?v=OwiQgAMqyUs",
        help="YouTube URL or worker-local video path.",
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=60.0)
    parser.add_argument("--model", default="weights/model_v1.1.0.pt")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--tracker", choices=("botsort.yaml", "bytetrack.yaml"), default="botsort.yaml")
    parser.add_argument("--match-config", default="data/match_config.json")
    parser.add_argument("--jersey-interval", type=int, default=24)
    parser.add_argument("--diagnostics-batch-size", type=int, default=100)
    parser.add_argument(
        "--no-jersey",
        action="store_true",
        help="Disable Jersey for a faster P2 event smoke run.",
    )
    args = parser.parse_args()
    if args.end <= args.start:
        parser.error("--end must be greater than --start")
    return args


if __name__ == "__main__":
    run_probe(parse_args())
