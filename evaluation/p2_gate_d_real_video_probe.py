from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_gate_d_report import build_gate_d_report

DEV_CLIPS = ("v1", "v2", "v3", "v4", "v5")
DEFAULT_NEUTRAL_MATCH_CONFIG = "evaluation/match_configs/unknown.json"


def _clip_defaults(clip: str) -> dict[str, Any]:
    if clip not in DEV_CLIPS:
        raise ValueError(f"Gate-D development clip must be one of {DEV_CLIPS}")
    path = PROJECT_ROOT / "evaluation" / "ground_truth" / f"{clip}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    video = dict(payload.get("video", {}) or {})
    match_config = PROJECT_ROOT / "evaluation" / "match_configs" / f"gate_c_{clip}.json"
    return {
        "video": str(video["source"]),
        "start": float(video.get("start_time_sec", 0.0) or 0.0),
        "end": float(video["end_time_sec"]),
        "match_config": str(match_config.relative_to(PROJECT_ROOT)) if match_config.exists() else DEFAULT_NEUTRAL_MATCH_CONFIG,
    }


def _install_gate_d_probe(
    analyzer_module: Any,
    p2_pipeline_cls: Any,
    gate_d_detector_cls: Any,
    *,
    clip: str,
) -> None:
    # Reuse the existing canonical P2 probe first, then add only Gate-D validation fields.
    from evaluation.p2_real_video_probe import _install_probe

    _install_probe(analyzer_module, p2_pipeline_cls)
    base_metrics_cls = analyzer_module.P2MetricsCollector
    base_attach_metrics = analyzer_module._attach_p2_metrics

    class GateDProbeP2MetricsCollector(base_metrics_cls):
        def observe(self, frame_state: Any) -> None:
            rows_before = len(self.state_rows)
            super().observe(frame_state)
            if len(self.state_rows) <= rows_before:
                return
            detector = dict(
                dict(getattr(frame_state, "diagnostics", {}) or {}).get("detector", {}) or {}
            )
            self.state_rows[-1]["game_player_predictions"] = [
                dict(row)
                for row in list(detector.get("game_player_predictions", []) or [])
                if isinstance(row, dict)
            ]
            self.state_rows[-1]["game_player_runtime"] = dict(
                detector.get("game_player_runtime", {}) or {}
            )

    def _attach_gate_d_artifacts(
        summary_path: Path,
        run_dir: Path,
        collector: Any,
    ) -> dict[str, Any]:
        p2_summary = base_attach_metrics(summary_path, run_dir, collector)
        report = build_gate_d_report(collector.state_rows, clip=clip)
        report_path = run_dir / "p2_gate_d_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["p2_gate_d_report_file"] = str(report_path)
        payload["p2_gate_d"] = {
            "clip": clip,
            "development_corpus": True,
            "exit_gate_unseen": False,
            "report_file": str(report_path),
            "simultaneous_identity_conflicts": report["invariants"][
                "simultaneous_same_game_player_conflicts"
            ],
            "resolved_observation_rate": report["resolved_observation_rate"],
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[P2_GATE_D_REPORT_FILE] {report_path.resolve()}", flush=True)
        return p2_summary

    analyzer_module.P2MetricsCollector = GateDProbeP2MetricsCollector
    analyzer_module._attach_p2_metrics = _attach_gate_d_artifacts
    analyzer_module.PerformanceDetector = gate_d_detector_cls


def run_probe(args: argparse.Namespace) -> None:
    defaults = _clip_defaults(args.clip)
    video = args.video or defaults["video"]
    start = defaults["start"] if args.start is None else float(args.start)
    end = defaults["end"] if args.end is None else float(args.end)
    match_config = args.match_config or defaults["match_config"]
    if end <= start:
        raise ValueError("--end must be greater than --start")

    os.environ["BASKETBALL_P2_ENABLED"] = "1"
    os.environ["BASKETBALL_SMART_BALL_MODE"] = "off"
    os.environ["BASKETBALL_SMART_BALL_PIPELINE"] = "0"
    os.environ["BASKETBALL_GAME_PLAYER_RESOLVER_V1"] = "1"

    # Delayed imports preserve lightweight CLI/report tooling and avoid loading the model
    # merely by importing this module.
    from services.analytics import analyzer
    from services.p2.pipeline import P2Pipeline
    from services.vision.game_player_gate_d_detector_v4 import PerformanceDetector as GateDDetector

    _install_gate_d_probe(analyzer, P2Pipeline, GateDDetector, clip=args.clip)
    print(
        "[P2_GATE_D_CLIP] "
        + json.dumps(
            {
                "clip": args.clip,
                "video": video,
                "start": start,
                "end": end,
                "match_config": match_config,
                "status": "challenge_non_blocking" if args.clip == "v2" else "development",
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    analyzer.main(
        video_source=video,
        start_time=start,
        end_time=end,
        model_path=args.model,
        detector_conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        tracker=args.tracker,
        match_config_path=match_config,
        enable_jersey=not args.no_jersey,
        jersey_interval_frames=args.jersey_interval,
        diagnostics_batch_size=args.diagnostics_batch_size,
        persist_results=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Gate-D game-player resolver on development clips V1-V5 only. "
            "V6/V7 are intentionally not addressable by this tool."
        )
    )
    parser.add_argument("--clip", required=True, choices=DEV_CLIPS)
    parser.add_argument("--video", help="Optional override for the frozen clip source.")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--model", default="weights/model_v1.1.0.pt")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument(
        "--tracker",
        choices=("botsort.yaml", "bytetrack.yaml"),
        default="botsort.yaml",
    )
    parser.add_argument("--match-config")
    parser.add_argument("--jersey-interval", type=int, default=24)
    parser.add_argument("--diagnostics-batch-size", type=int, default=100)
    parser.add_argument("--no-jersey", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_probe(parse_args())
