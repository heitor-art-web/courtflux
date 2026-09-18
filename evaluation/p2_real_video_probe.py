from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NEUTRAL_MATCH_CONFIG = "evaluation/match_configs/unknown.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_probe(analyzer_module: Any, p2_pipeline_cls: Any) -> None:
    """Patch only probe-specific artifact export around the canonical analyzer.

    The analyzer owns the single canonical ``services.p2.metrics.P2MetricsCollector``
    instance and writes ``summary.p2`` plus ``p2_metrics.json``. The probe only adds the
    raw event/state JSONL artifacts from that same collector; it must not implement or
    instantiate a second metrics collector.

    For validation diagnostics only, the probe wraps the analyzer's collector class so
    each exported state row can include the shot tracker's current start-gate inputs and
    Identity-v2 team prediction evidence. The canonical collector implementation and
    basketball decisions remain unchanged.
    """

    base_attach_p2_metrics = analyzer_module._attach_p2_metrics
    base_metrics_cls = analyzer_module.P2MetricsCollector

    class ProbeP2MetricsCollector(base_metrics_cls):
        def observe(self, frame_state: Any) -> None:
            rows_before = len(self.state_rows)
            super().observe(frame_state)
            if len(self.state_rows) <= rows_before:
                return

            diagnostics = dict(getattr(frame_state, "diagnostics", {}) or {})
            p2 = dict(diagnostics.get("p2", {}) or {})
            shot = dict(p2.get("shot", {}) or {})
            start_gate = dict(shot.get("start_gate_diagnostics", {}) or {})
            detector = dict(diagnostics.get("detector", {}) or {})
            team_predictions = [
                dict(row)
                for row in list(detector.get("team_predictions", []) or [])
                if isinstance(row, dict)
            ]
            team_runtime = dict(detector.get("team_runtime", {}) or {})

            self.state_rows[-1]["shot_start_gate"] = start_gate
            self.state_rows[-1]["team_predictions"] = team_predictions
            self.state_rows[-1]["team_runtime"] = team_runtime

    def _attach_probe_artifacts(
        summary_path: Path,
        run_dir: Path,
        collector: Any,
    ) -> dict[str, Any]:
        p2_summary = base_attach_p2_metrics(summary_path, run_dir, collector)

        events_path = run_dir / "p2_events.jsonl"
        events_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in collector.event_rows
            ),
            encoding="utf-8",
        )
        states_path = run_dir / "p2_states.jsonl"
        states_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in collector.state_rows
            ),
            encoding="utf-8",
        )

        print(f"[P2_EVENTS_FILE] {events_path.resolve()}", flush=True)
        print(f"[P2_STATES_FILE] {states_path.resolve()}", flush=True)
        return p2_summary

    analyzer_module.P2MetricsCollector = ProbeP2MetricsCollector
    analyzer_module._attach_p2_metrics = _attach_probe_artifacts
    analyzer_module.Pipeline = p2_pipeline_cls


def run_probe(args: argparse.Namespace) -> None:
    # Freeze the rebuilt-P2 validation contract before runtime construction. Smart Ball
    # stays explicitly OFF so comparisons do not silently inherit an experiment default.
    os.environ["BASKETBALL_P2_ENABLED"] = "1"
    os.environ["BASKETBALL_SMART_BALL_MODE"] = "off"
    os.environ["BASKETBALL_SMART_BALL_PIPELINE"] = "0"

    # Heavy runtime imports remain delayed so importing this probe does not require the
    # full OpenCV/Ultralytics runtime in lightweight tooling.
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
    parser.add_argument(
        "--tracker",
        choices=("botsort.yaml", "bytetrack.yaml"),
        default="botsort.yaml",
    )
    parser.add_argument(
        "--match-config",
        default=DEFAULT_NEUTRAL_MATCH_CONFIG,
        help=(
            "Match config used for team evidence. Defaults to the neutral/unknown P2 GT "
            "config; pass data/match_config.json explicitly for the Porto/Ovarense known-team control."
        ),
    )
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
