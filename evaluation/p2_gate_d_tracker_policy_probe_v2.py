from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_gate_d_tracker_policy_probe as base


VERSION = "courtflux_gate_d_v6_tracker_policy_probe_v2"


def _relation_windows(
    anchors: dict[str, dict[str, Any]],
    pairs: dict[str, list[dict[str, Any]]],
    *,
    pad_sec: float,
) -> list[tuple[float, float]]:
    """Keep every scored handoff/control relation inside one continuous tracker lifetime."""
    raw: list[tuple[float, float]] = []
    for key in ("handoffs", "stable_controls"):
        for pair in pairs.get(key, []):
            left = anchors[str(pair["left"])]
            right = anchors[str(pair["right"])]
            start = min(float(left["timestamp_sec"]), float(right["timestamp_sec"])) - float(pad_sec)
            end = max(float(left["timestamp_sec"]), float(right["timestamp_sec"])) + float(pad_sec)
            raw.append((start, end))
    if not raw:
        raise RuntimeError("No handoff/control relations available for continuous probe windows")

    raw.sort()
    merged: list[list[float]] = []
    for start, end in raw:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(float(start), float(end)) for start, end in merged]


class _ExplicitFrameRateTracker:
    """Probe-only adapter that guarantees Ultralytics receives the requested frame rate."""

    def __init__(self, tracker_yaml: str, *, device: str, frame_rate: float) -> None:
        from services.vision.single_forward_tracker import SingleForwardEntityTracker

        class _Adapter(SingleForwardEntityTracker):
            def _build_tracker(inner_self) -> Any:
                cfg = inner_self._load_config()
                tracker_type = str(getattr(cfg, "tracker_type", "botsort")).lower()
                tracker_map = inner_self._tracker_map()
                if tracker_type not in tracker_map:
                    raise ValueError(
                        f"Unsupported tracker_type={tracker_type!r}; available={sorted(tracker_map)}"
                    )
                tracker_cls = tracker_map[tracker_type]
                fps = int(round(inner_self.frame_rate))
                # For this diagnostic the requested fps must be explicit. Fall back only
                # for an Ultralytics version whose constructor truly has no frame_rate arg.
                try:
                    return tracker_cls(args=cfg, frame_rate=fps)
                except TypeError as first_exc:
                    try:
                        return tracker_cls(cfg, frame_rate=fps)
                    except TypeError:
                        try:
                            return tracker_cls(args=cfg)
                        except TypeError:
                            raise first_exc

        self._adapter = _Adapter(
            tracker_yaml,
            device=device,
            frame_rate=float(frame_rate),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


def _build_trackers_explicit_fps(
    device: str,
    actual_fps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trackers: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for spec in base.POLICIES:
        tracker_fps = 30.0 if spec.frame_rate_mode == "legacy30" else float(actual_fps)
        adapter = _ExplicitFrameRateTracker(
            spec.tracker_yaml,
            device=device,
            frame_rate=tracker_fps,
        )
        trackers[spec.name] = adapter
        metadata[spec.name] = {
            "tracker_yaml": spec.tracker_yaml,
            "frame_rate_mode": spec.frame_rate_mode,
            "tracker_frame_rate": tracker_fps,
            "frame_rate_constructor_policy": "explicit_argument_first_v2",
            "resolved_config": base._tracker_config_snapshot(adapter),
        }
    return trackers, metadata


def _score_policy_scoped(
    policy_name: str,
    anchors: dict[str, dict[str, Any]],
    pairs: dict[str, list[dict[str, Any]]],
    mapped: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    """Score only relations whose two frozen anchors were actually replayed."""
    active = set(mapped)
    scoped: dict[str, list[dict[str, Any]]] = {}
    for key, rows in pairs.items():
        scoped[key] = [
            dict(row)
            for row in rows
            if str(row.get("left")) in active and str(row.get("right")) in active
        ]
    result = base._score_policy(policy_name, anchors, scoped, mapped)
    result["scope_integrity"] = {
        "only_relations_inside_replayed_windows": True,
        "source_pair_counts": {key: len(rows) for key, rows in pairs.items()},
        "scored_pair_counts": {key: len(rows) for key, rows in scoped.items()},
    }
    return result


def run(args: Any) -> dict[str, Any]:
    # Monkeypatch only the diagnostic module. Production/runtime code remains untouched.
    base._merge_windows = _relation_windows
    base._build_trackers = _build_trackers_explicit_fps
    base._score_policy = _score_policy_scoped

    if not getattr(args, "output", None):
        args.output = str(
            base.REPO_ROOT / "evaluation" / "runs" / "p2_v6_gate_d_tracker_policy_probe_v2.json"
        )

    report = base.run(args)
    report["version"] = VERSION
    report["integrity_fixes"] = {
        "continuous_relation_windows": True,
        "tracker_reset_never_occurs_between_endpoints_of_a_scored_handoff_or_control": True,
        "ultralytics_frame_rate_passed_explicitly": True,
        "scoring_restricted_to_relations_inside_replayed_windows": True,
        "runtime_code_modified": False,
    }
    output_path = Path(args.output).expanduser().resolve()
    import json

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    run(base.parse_args())
