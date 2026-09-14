from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_signature(summary: dict[str, Any]) -> dict[str, Any]:
    runtime = summary.get("runtime") or {}
    return {
        "video": runtime.get("video"),
        "detector": runtime.get("detector"),
        "tracker": runtime.get("tracker"),
        "imgsz": runtime.get("imgsz"),
        "tracker_input_confidence": runtime.get("tracker_input_confidence"),
        "start_sec": runtime.get("start_sec"),
        "end_sec": runtime.get("end_sec"),
    }


def _assert_runtime_match(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    sig_a = _runtime_signature(a)
    sig_b = _runtime_signature(b)
    mismatches = {
        key: {"best_temporal": sig_a.get(key), "vit_b_uncertainty": sig_b.get(key)}
        for key in sig_a
        if sig_a.get(key) != sig_b.get(key)
    }
    if mismatches:
        raise ValueError(
            "Refusing jersey comparison because runtime contracts differ: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return sig_a


def _distribution_items(value: Any) -> list[list[Any]]:
    if isinstance(value, dict):
        return [[str(key), int(count)] for key, count in sorted(value.items(), key=lambda item: (-int(item[1]), str(item[0])))]
    if isinstance(value, list):
        return [[str(row[0]), int(row[1])] for row in value if isinstance(row, (list, tuple)) and len(row) >= 2]
    return []


def _normalize_temporal(summary: dict[str, Any]) -> dict[str, Any]:
    tracking = summary.get("tracking") or {}
    jersey = summary.get("jersey_operational") or {}
    return {
        "recognizer": "best_temporal",
        "model": (summary.get("runtime") or {}).get("jersey_model"),
        "tracked_player_ids": tracking.get("tracked_player_ids"),
        "short_track_proxy_rate": tracking.get("short_track_proxy_rate"),
        "predicted_tracks": jersey.get("predicted_tracks"),
        "unknown_tracks": jersey.get("unknown_tracks"),
        "operational_coverage": jersey.get("prediction_coverage"),
        "policy_evaluable_tracks": None,
        "full_support_coverage": jersey.get("full_support_coverage"),
        "multi_sequence_stability_rate": jersey.get("multi_sequence_stability_rate"),
        "median_modal_fraction": jersey.get("median_modal_fraction"),
        "prediction_distribution": _distribution_items(jersey.get("prediction_distribution")),
        "accuracy_claim": False,
    }


def _normalize_uncertainty(summary: dict[str, Any]) -> dict[str, Any]:
    jersey = summary.get("jersey") or {}
    stability = jersey.get("accepted_track_stability") or {}
    short = jersey.get("short_track_proxy") or {}
    return {
        "recognizer": "vit_b_uncertainty",
        "model": (summary.get("runtime") or {}).get("jersey_model"),
        "tracked_player_ids": jersey.get("tracked_player_ids"),
        "short_track_proxy_rate": short.get("rate"),
        "predicted_tracks": jersey.get("accepted_tracks"),
        "unknown_tracks": jersey.get("unknown_tracks"),
        "operational_coverage": jersey.get("operational_acceptance_rate"),
        "policy_evaluable_tracks": jersey.get("policy_evaluable_tracks"),
        "full_support_coverage": None,
        "multi_sequence_stability_rate": None,
        "median_modal_fraction": stability.get("median_modal_frame_fraction"),
        "prediction_distribution": _distribution_items(jersey.get("accepted_number_distribution")),
        "accuracy_claim": False,
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _top_distribution(rows: list[list[Any]], limit: int = 8) -> str:
    if not rows:
        return "n/a"
    return ", ".join(f"{number}:{count}" for number, count in rows[:limit])


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    t = report["recognizers"]["best_temporal"]
    u = report["recognizers"]["vit_b_uncertainty"]
    lines = [
        "# Jersey recognizer operational comparison — P1 v1.1 tracks",
        "",
        "No human jersey ground truth is used. This comparison is limited to operational coverage, abstention, support and temporal stability. It does **not** measure jersey accuracy or precision.",
        "",
        "Both recognizers are compared only after their runtime contracts match on video, detector, tracker, tracker confidence, image size and time range.",
        "",
        "| Metric | best_temporal | ViT-B uncertainty |",
        "| --- | ---: | ---: |",
        f"| tracked player IDs | {t['tracked_player_ids']} | {u['tracked_player_ids']} |",
        f"| short-track proxy rate | {_pct(t['short_track_proxy_rate'])} | {_pct(u['short_track_proxy_rate'])} |",
        f"| tracks with concrete jersey hypothesis | {t['predicted_tracks']} | {u['predicted_tracks']} |",
        f"| UNKNOWN tracks | {t['unknown_tracks']} | {u['unknown_tracks']} |",
        f"| operational coverage | {_pct(t['operational_coverage'])} | {_pct(u['operational_coverage'])} |",
        f"| policy-evaluable tracks | n/a | {u['policy_evaluable_tracks']} |",
        f"| 8-distinct-crop full-support coverage | {_pct(t['full_support_coverage'])} | n/a |",
        f"| multi-sequence stability | {_pct(t['multi_sequence_stability_rate'])} | n/a |",
        f"| median modal fraction | {_pct(t['median_modal_fraction'])} | {_pct(u['median_modal_fraction'])} |",
        "",
        "## Prediction concentration",
        "",
        f"- best_temporal: `{_top_distribution(t['prediction_distribution'])}`",
        f"- ViT-B uncertainty: `{_top_distribution(u['prediction_distribution'])}`",
        "",
        "## Interpretation boundary",
        "",
        "- A recognizer with higher operational coverage is not necessarily more accurate.",
        "- Repeated-crop temporal sequences remain low-support diagnostics, not full-support evidence.",
        "- ViT-B accepted numbers remain hypotheses produced by the frozen #37 policy.",
        "- Human GT is required before any promotion claim based on jersey correctness.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def compare(temporal: dict[str, Any], uncertainty: dict[str, Any]) -> dict[str, Any]:
    runtime = _assert_runtime_match(temporal, uncertainty)
    return {
        "purpose": "P1 Jersey recognizer operational comparison on matched v1.1 tracking runtime",
        "evidence_status": "no_ground_truth_operational_comparison_only",
        "runtime_contract": runtime,
        "recognizers": {
            "best_temporal": _normalize_temporal(temporal),
            "vit_b_uncertainty": _normalize_uncertainty(uncertainty),
        },
        "guardrails": {
            "ground_truth_used": False,
            "accuracy_claim": False,
            "precision_claim": False,
            "runtime_match_required": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare best_temporal and ViT-B jersey summaries under the same P1 v1.1 runtime")
    parser.add_argument("--temporal-summary", type=Path, required=True)
    parser.add_argument("--uncertainty-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    temporal_path = args.temporal_summary.expanduser().resolve()
    uncertainty_path = args.uncertainty_summary.expanduser().resolve()
    for path in (temporal_path, uncertainty_path):
        if not path.exists():
            raise FileNotFoundError(path)

    report = compare(_load(temporal_path), _load(uncertainty_path))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "jersey_model_comparison.json"
    md_path = output_dir / "jersey_model_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"comparison={md_path}")
    print(f"comparison_json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
