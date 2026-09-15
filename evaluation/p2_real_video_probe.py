from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


class P2MetricsCollector:
    """Aggregate exploratory P2 metrics without changing runtime decisions."""

    def __init__(self) -> None:
        self.frames_seen = 0
        self.metric_eligible_frames = 0
        self.possession_locked_frames = 0
        self.possession_confidence_total = 0.0
        self.possession_confidence_samples = 0
        self.p1_not_mutated_frames = 0
        self.event_counts: Counter[str] = Counter()
        self.decision_counts: Counter[str] = Counter()
        self.shot_outcomes: Counter[str] = Counter()
        self.event_confidence_total: Counter[str] = Counter()
        self.event_confidence_samples: Counter[str] = Counter()
        self.events_with_actor = 0
        self.events_with_team = 0
        self.total_events = 0
        self.event_rows: list[dict[str, Any]] = []

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / max(denominator, 1), 2)

    def observe(self, frame_state: Any) -> None:
        self.frames_seen += 1
        metric_eligible = bool(getattr(frame_state, "valid_gameplay", False))
        if metric_eligible:
            self.metric_eligible_frames += 1

        diagnostics = dict(getattr(frame_state, "diagnostics", {}) or {})
        p2 = dict(diagnostics.get("p2", {}) or {})
        if not p2:
            return

        if bool(p2.get("p1_state_is_not_mutated", False)):
            self.p1_not_mutated_frames += 1

        possession = dict(p2.get("possession", {}) or {})
        current_track_id = possession.get("current_track_id")
        if metric_eligible and current_track_id is not None:
            self.possession_locked_frames += 1
            try:
                confidence = float(possession.get("current_confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            self.possession_confidence_total += confidence
            self.possession_confidence_samples += 1

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
            if row.get("team_id") not in {None, "", "unknown", "UNKNOWN"}:
                self.events_with_team += 1

            if event_type == "shot_attempt":
                attributes = dict(row.get("attributes", {}) or {})
                self.shot_outcomes[str(attributes.get("outcome") or "unknown")] += 1

            self.event_rows.append(row)

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
        shot_attempts = int(self.event_counts.get("shot_attempt", 0))
        unknown_shots = int(self.shot_outcomes.get("unknown", 0))

        return {
            "version": "p2_real_video_probe_v1",
            "frames_seen": self.frames_seen,
            "metric_eligible_frames": self.metric_eligible_frames,
            "possession": {
                "locked_frames": self.possession_locked_frames,
                "coverage_rate": self._rate(
                    self.possession_locked_frames,
                    self.metric_eligible_frames,
                ),
                "unknown_rate": self._rate(
                    max(0, self.metric_eligible_frames - self.possession_locked_frames),
                    self.metric_eligible_frames,
                ),
                "avg_lock_confidence": (
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
            },
            "shots": {
                "attempts": shot_attempts,
                "outcomes": dict(sorted(self.shot_outcomes.items())),
                "unknown_outcome_rate": self._rate(unknown_shots, shot_attempts),
            },
            "passes": int(self.event_counts.get("pass_candidate", 0)),
            "turnovers": int(self.event_counts.get("turnover_candidate", 0)),
            "rebounds": int(self.event_counts.get("rebound_candidate", 0)),
            "p1_state_not_mutated_rate": self._rate(
                self.p1_not_mutated_frames,
                self.frames_seen,
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
            print(f"[P2_METRICS] {json.dumps(p2_summary, separators=(',', ':'))}", flush=True)
            print(f"[P2_METRICS_FILE] {metrics_path.resolve()}", flush=True)
            print(f"[P2_EVENTS_FILE] {events_path.resolve()}", flush=True)
            return path

    analyzer_module.RunDiagnostics = P2ProbeDiagnostics
    analyzer_module.Pipeline = p2_pipeline_cls


def run_probe(args: argparse.Namespace) -> None:
    # Heavy runtime imports are deliberately delayed so P2 metric aggregation remains
    # unit-testable in lightweight CI without OpenCV/Ultralytics/weights.
    from services.analytics import analyzer
    from services.p2.pipeline import P2Pipeline

    os.environ["BASKETBALL_P2_ENABLED"] = "1"
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
        detection_only=True,
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
