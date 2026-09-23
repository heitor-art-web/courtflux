from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.p2_gate_d_real_video_probe import _clip_defaults, _install_gate_d_probe


class _BaseCollector:
    def __init__(self) -> None:
        self.state_rows = []
        self.event_rows = []

    def observe(self, frame_state) -> None:
        self.state_rows.append(
            {
                "frame_id": int(frame_state.frame_id),
                "timestamp_sec": float(frame_state.timestamp),
            }
        )


def _fake_probe_module() -> ModuleType:
    module = ModuleType("evaluation.p2_real_video_probe")

    def _install_probe(analyzer_module, p2_pipeline_cls) -> None:
        analyzer_module.P2MetricsCollector = _BaseCollector
        analyzer_module.Pipeline = p2_pipeline_cls

        def _attach(summary_path: Path, run_dir: Path, collector):
            if not summary_path.exists():
                summary_path.write_text("{}\n", encoding="utf-8")
            return {"frames": len(collector.state_rows)}

        analyzer_module._attach_p2_metrics = _attach

    module._install_probe = _install_probe
    return module


def test_gate_d_probe_exports_predictions_and_report() -> None:
    analyzer = SimpleNamespace(P2MetricsCollector=_BaseCollector, _attach_p2_metrics=None)
    fake_detector = object()
    with patch.dict(sys.modules, {"evaluation.p2_real_video_probe": _fake_probe_module()}):
        _install_gate_d_probe(analyzer, object, fake_detector, clip="v4")

    collector = analyzer.P2MetricsCollector()
    collector.observe(
        SimpleNamespace(
            frame_id=7,
            timestamp=0.7,
            diagnostics={
                "detector": {
                    "game_player_predictions": [
                        {
                            "track_id": 11,
                            "episode_index": 0,
                            "game_player_id": "game_player_001",
                            "appearance_available": True,
                        }
                    ],
                    "game_player_runtime": {"version": "game_player_resolver_v2"},
                }
            },
        )
    )
    assert collector.state_rows[-1]["game_player_predictions"][0]["track_id"] == 11
    assert analyzer.PerformanceDetector is fake_detector

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        summary_path = run_dir / "summary.json"
        summary_path.write_text("{}\n", encoding="utf-8")
        analyzer._attach_p2_metrics(summary_path, run_dir, collector)
        report = json.loads((run_dir / "p2_gate_d_report.json").read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert report["clip"] == "v4"
        assert report["resolved_observations"] == 1
        assert summary["p2_gate_d"]["clip"] == "v4"


def test_exit_gate_clips_are_not_addressable() -> None:
    try:
        _clip_defaults("v6")
    except ValueError:
        return
    raise AssertionError("V6 must not be addressable by the Gate-D development probe")


def main() -> None:
    test_gate_d_probe_exports_predictions_and_report()
    test_exit_gate_clips_are_not_addressable()
    print("p2_gate_d_real_video_probe_test: PASS")


if __name__ == "__main__":
    main()
