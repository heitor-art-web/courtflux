from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GT = REPO_ROOT / "evaluation" / "ground_truth" / "gate_d_v6_human_ball_carrier_gt_v1.json"
DEFAULT_GEOMETRY = REPO_ROOT / "evaluation" / "ground_truth" / "gate_d_v6_tracker_anchor_geometry_v1.json"
DEFAULT_V6 = REPO_ROOT / "evaluation" / "ground_truth" / "v6.json"
DEFAULT_MODEL = REPO_ROOT / "weights" / "model_v1.1.0.pt"
DEFAULT_MATCH_CONFIG = REPO_ROOT / "evaluation" / "match_configs" / "unknown.json"

MAX_LOCAL_GAP_SEC = 10.0
WINDOW_PAD_SEC = 2.0


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    tracker_yaml: str
    frame_rate_mode: str  # "legacy30" or "actual"


POLICIES = (
    PolicySpec("botsort_legacy30", "botsort.yaml", "legacy30"),
    PolicySpec("botsort_actual_fps", "botsort.yaml", "actual"),
    PolicySpec("bytetrack_legacy30", "bytetrack.yaml", "legacy30"),
    PolicySpec("bytetrack_actual_fps", "bytetrack.yaml", "actual"),
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(v) for v in left]
    rx1, ry1, rx2, ry2 = [float(v) for v in right]
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    la = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = la + ra - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _center_distance_norm(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(v) for v in left]
    rx1, ry1, rx2, ry2 = [float(v) for v in right]
    lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
    rcx, rcy = (rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0
    scale = max(1.0, math.hypot(lx2 - lx1, ly2 - ly1), math.hypot(rx2 - rx1, ry2 - ry1))
    return float(math.hypot(lcx - rcx, lcy - rcy) / scale)


def _match_anchor(
    frozen_box: list[float],
    tracked_rows: list[dict[str, Any]],
    *,
    min_iou: float,
    max_center_distance_norm: float,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for row in tracked_rows:
        box = row.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        box_f = [float(v) for v in box]
        iou = _iou(frozen_box, box_f)
        center = _center_distance_norm(frozen_box, box_f)
        # IoU is primary. Center distance is a fallback for Kalman boxes that expand/shrink.
        if iou < float(min_iou) and center > float(max_center_distance_norm):
            continue
        candidates.append((iou, -center, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    iou, neg_center, row = candidates[0]
    return {
        "track_id": int(row.get("track_id", -1)),
        "box": [float(v) for v in row.get("box")],
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "iou_to_frozen_anchor": round(float(iou), 6),
        "center_distance_norm": round(float(-neg_center), 6),
    }


def _anchors_by_id(geometry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in geometry.get("anchors") or []:
        if not isinstance(row, dict):
            continue
        anchor_id = str(row.get("id") or "")
        if not anchor_id:
            continue
        result[anchor_id] = dict(row)
    return result


def _validate_geometry_against_gt(
    gt: dict[str, Any],
    anchors: dict[str, dict[str, Any]],
) -> None:
    mapped = {
        str(row.get("id") or ""): row
        for row in (gt.get("mapped_observations") or [])
        if isinstance(row, dict) and row.get("id")
    }
    missing = sorted(set(anchors) - set(mapped))
    if missing:
        raise ValueError("Tracker anchor geometry contains IDs absent from frozen human GT: " + ", ".join(missing))

    mismatches: list[str] = []
    for anchor_id, anchor in anchors.items():
        gt_row = mapped[anchor_id]
        endpoint = dict(gt_row.get("endpoint") or {})
        checks = {
            "human_player": (str(anchor.get("human_player")), str(gt_row.get("human_player"))),
            "team_id": (str(anchor.get("team_id")), str(gt_row.get("team_id"))),
            "frame_id": (int(anchor.get("frame_id", -1)), int(endpoint.get("frame_id", -2))),
            "baseline_track_id": (
                int(anchor.get("baseline_track_id", -1)),
                int(endpoint.get("track_id", -2)),
            ),
        }
        for field, (geometry_value, gt_value) in checks.items():
            if geometry_value != gt_value:
                mismatches.append(
                    f"{anchor_id}.{field}: geometry={geometry_value!r} gt={gt_value!r}"
                )
    if mismatches:
        raise ValueError(
            "Tracker anchor geometry no longer matches frozen human GT: "
            + "; ".join(mismatches)
        )


def _derive_probe_pairs(
    anchors: dict[str, dict[str, Any]],
    *,
    max_gap_sec: float = MAX_LOCAL_GAP_SEC,
) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        anchors.values(),
        key=lambda row: (float(row["timestamp_sec"]), str(row["id"])),
    )
    handoffs: list[dict[str, Any]] = []
    stable_controls: list[dict[str, Any]] = []
    nearby_different: list[dict[str, Any]] = []
    nearby_same: list[dict[str, Any]] = []

    for i, left in enumerate(ordered):
        for right in ordered[i + 1 :]:
            gap = float(right["timestamp_sec"]) - float(left["timestamp_sec"])
            if gap > float(max_gap_sec):
                break
            same_human = str(left["human_player"]) == str(right["human_player"])
            same_baseline_track = int(left["baseline_track_id"]) == int(right["baseline_track_id"])
            pair = {
                "left": str(left["id"]),
                "right": str(right["id"]),
                "left_human": str(left["human_player"]),
                "right_human": str(right["human_player"]),
                "gap_sec": round(float(gap), 6),
                "baseline_track_id": (
                    int(left["baseline_track_id"]) if same_baseline_track else None
                ),
            }
            if same_human:
                nearby_same.append(dict(pair))
            else:
                nearby_different.append(dict(pair))

            if same_baseline_track and not same_human:
                handoffs.append(dict(pair))

            if same_baseline_track and same_human:
                # A SAME endpoint pair is not a clean tracker-stability control if the same
                # raw track is known to carry a different human in between the endpoints.
                contaminated_between = any(
                    int(mid["baseline_track_id"]) == int(left["baseline_track_id"])
                    and float(left["timestamp_sec"]) < float(mid["timestamp_sec"]) < float(right["timestamp_sec"])
                    and str(mid["human_player"]) != str(left["human_player"])
                    for mid in ordered
                )
                if not contaminated_between:
                    stable_controls.append(dict(pair))

    return {
        "handoffs": handoffs,
        "stable_controls": stable_controls,
        "nearby_different": nearby_different,
        "nearby_same": nearby_same,
    }


def _merge_windows(
    anchors: dict[str, dict[str, Any]],
    pairs: dict[str, list[dict[str, Any]]],
    *,
    pad_sec: float,
) -> list[tuple[float, float]]:
    required_ids: set[str] = set()
    for key in ("handoffs", "stable_controls"):
        for pair in pairs.get(key, []):
            required_ids.add(str(pair["left"]))
            required_ids.add(str(pair["right"]))
    if not required_ids:
        raise RuntimeError("No handoff/control anchors were derived from frozen geometry")

    raw: list[tuple[float, float]] = []
    for anchor_id in sorted(required_ids):
        ts = float(anchors[anchor_id]["timestamp_sec"])
        raw.append((ts - float(pad_sec), ts + float(pad_sec)))
    raw.sort()

    merged: list[list[float]] = []
    for start, end in raw:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(float(start), float(end)) for start, end in merged]


def _resolve_video(v6_payload: dict[str, Any], override: str | None) -> str:
    if override:
        return str(Path(override).expanduser().resolve())
    video = dict(v6_payload.get("video") or {})
    youtube_id = str(video.get("source") or "")
    local = REPO_ROOT / "data" / "VLZugJeVOSU.mp4"
    if local.exists():
        return str(local)
    from services.ingestion.youtube_downloader import YouTubeDownloader

    return str(YouTubeDownloader(output_dir="data").get_local_video(youtube_id))


def _tracker_config_snapshot(adapter: Any) -> dict[str, Any]:
    try:
        cfg = adapter._load_config()
        raw = vars(cfg)
    except Exception:
        raw = {}
    keys = (
        "tracker_type",
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
        "fuse_score",
        "gmc_method",
        "proximity_thresh",
        "appearance_thresh",
        "with_reid",
        "model",
    )
    return {key: raw.get(key) for key in keys if key in raw}


def _build_trackers(device: str, actual_fps: float) -> tuple[dict[str, Any], dict[str, Any]]:
    from services.vision.single_forward_tracker import SingleForwardEntityTracker

    trackers: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for spec in POLICIES:
        tracker_fps = 30.0 if spec.frame_rate_mode == "legacy30" else float(actual_fps)
        adapter = SingleForwardEntityTracker(
            spec.tracker_yaml,
            device=device,
            frame_rate=tracker_fps,
        )
        trackers[spec.name] = adapter
        metadata[spec.name] = {
            "tracker_yaml": spec.tracker_yaml,
            "frame_rate_mode": spec.frame_rate_mode,
            "tracker_frame_rate": tracker_fps,
            "resolved_config": _tracker_config_snapshot(adapter),
        }
    return trackers, metadata


def _prepare_detector(model_path: Path, device: str, conf: float, imgsz: int) -> Any:
    from ultralytics import YOLO
    from services.vision.game_player_gate_d_detector_v6 import (
        PerformanceDetector as GateDDetector,
    )
    from services.vision.team_only_analyzer import TeamOnlyAnalyzer

    visual = TeamOnlyAnalyzer(profiling_path=str(DEFAULT_MATCH_CONFIG))
    model = YOLO(str(model_path))
    model.to(device)
    return GateDDetector(
        model=model,
        visual_analyzer=visual,
        profiling_path=str(DEFAULT_MATCH_CONFIG),
        device=device,
        detector_conf=float(conf),
        imgsz=int(imgsz),
        iou=0.5,
        tracker="botsort.yaml",
        max_det=40,
        enable_jersey=False,
        jersey_interval_frames=24,
    )


def _entity_boxes(detector: Any, frame: Any) -> Any:
    results = detector.model.predict(
        frame,
        conf=detector.inference_conf_thresh,
        iou=detector.iou,
        imgsz=detector.imgsz,
        max_det=detector.ball_candidate_max_det,
        device=detector.device,
        verbose=False,
    )
    result = results[0] if results else None
    if result is None:
        return None
    cpu_boxes, raw_rows = detector._bulk_raw_rows(result)
    tracked_ids = detector._tracked_class_ids_for_single_forward()
    entity_indices = [
        row["detection_index"]
        for row in raw_rows
        if int(row["class_id"]) in tracked_ids
        and float(row["confidence"]) >= detector.player_conf_thresh
    ]
    if cpu_boxes is None:
        return None
    return cpu_boxes[entity_indices] if entity_indices else cpu_boxes[:0]


def _reset_trackers(trackers: dict[str, Any]) -> None:
    for tracker in trackers.values():
        tracker.reset()


def _score_policy(
    policy_name: str,
    anchors: dict[str, dict[str, Any]],
    pairs: dict[str, list[dict[str, Any]]],
    mapped: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    def relation_rows(source: list[dict[str, Any]], expected_same: bool) -> tuple[int, int, int]:
        resolved = 0
        correct = 0
        false_same = 0
        for pair in source:
            left = mapped.get(str(pair["left"]))
            right = mapped.get(str(pair["right"]))
            if not left or not right:
                continue
            resolved += 1
            is_same = int(left["track_id"]) == int(right["track_id"])
            if is_same == expected_same:
                correct += 1
            if not expected_same and is_same:
                false_same += 1
        return resolved, correct, false_same

    handoff_resolved, handoff_correct, _ = relation_rows(pairs["handoffs"], False)
    control_resolved, control_correct, _ = relation_rows(pairs["stable_controls"], True)
    diff_resolved, diff_correct, diff_false_same = relation_rows(pairs["nearby_different"], False)
    same_resolved, same_correct, _ = relation_rows(pairs["nearby_same"], True)

    # Every anchor inside the short probe windows participates in the nearby-pair audit,
    # so coverage is measured over the entire mapped set, not just handoff/control endpoints.
    required_anchor_ids = sorted(mapped)
    anchor_resolved = sum(1 for key in required_anchor_ids if mapped.get(key) is not None)

    return {
        "policy": policy_name,
        "required_anchor_coverage": {
            "resolved": anchor_resolved,
            "total": len(required_anchor_ids),
            "rate": anchor_resolved / len(required_anchor_ids) if required_anchor_ids else 0.0,
        },
        "handoff_split_recall": {
            "correct": handoff_correct,
            "resolved": handoff_resolved,
            "total": len(pairs["handoffs"]),
            "rate": handoff_correct / len(pairs["handoffs"]) if pairs["handoffs"] else 0.0,
        },
        "stable_control_continuity": {
            "correct": control_correct,
            "resolved": control_resolved,
            "total": len(pairs["stable_controls"]),
            "rate": control_correct / len(pairs["stable_controls"]) if pairs["stable_controls"] else 0.0,
        },
        "nearby_different_false_merge": {
            "false_merges": diff_false_same,
            "resolved": diff_resolved,
            "total": len(pairs["nearby_different"]),
            "rate_on_all_pairs": (
                diff_false_same / len(pairs["nearby_different"])
                if pairs["nearby_different"]
                else 0.0
            ),
            "rate_on_resolved_pairs": diff_false_same / diff_resolved if diff_resolved else None,
        },
        "nearby_same_raw_continuity": {
            "correct": same_correct,
            "resolved": same_resolved,
            "total": len(pairs["nearby_same"]),
            "rate_on_all_pairs": same_correct / len(pairs["nearby_same"]) if pairs["nearby_same"] else 0.0,
        },
        "promising_for_development_followup": bool(
            len(pairs["handoffs"]) > 0
            and handoff_correct == len(pairs["handoffs"])
            and len(pairs["stable_controls"]) > 0
            and control_correct == len(pairs["stable_controls"])
            and diff_false_same == 0
            and anchor_resolved == len(required_anchor_ids)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt = _load_json(Path(args.gt).expanduser().resolve())
    geometry = _load_json(Path(args.geometry).expanduser().resolve())
    v6 = _load_json(DEFAULT_V6)
    anchors = _anchors_by_id(geometry)
    _validate_geometry_against_gt(gt, anchors)
    pairs = _derive_probe_pairs(anchors, max_gap_sec=float(args.max_gap_sec))
    windows = _merge_windows(anchors, pairs, pad_sec=float(args.pad_sec))

    if not pairs["handoffs"]:
        raise RuntimeError("Frozen geometry did not yield any known raw-track handoffs")
    if not pairs["stable_controls"]:
        raise RuntimeError("Frozen geometry did not yield any clean SAME raw-track controls")

    video_path = _resolve_video(v6, args.video)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if actual_fps <= 1.0:
        actual_fps = float(geometry.get("video", {}).get("fps", 60.0) or 60.0)

    from services.analytics.analyzer import resolve_device
    device = resolve_device(args.device)

    print(
        "[Tracker policy probe] "
        f"video_fps={actual_fps:.3f} windows={[(round(a,3), round(b,3)) for a,b in windows]} "
        f"handoffs={len(pairs['handoffs'])} controls={len(pairs['stable_controls'])}",
        flush=True,
    )
    print("[Tracker policy probe] loading detector once; tracker policies share every YOLO forward", flush=True)
    detector = _prepare_detector(model_path, device, args.conf, args.imgsz)
    trackers, policy_metadata = _build_trackers(device, actual_fps)

    # For every frozen anchor, keep the closest decoded frame snapshot per policy.
    snapshots: dict[str, dict[str, dict[str, Any]]] = {
        policy.name: {} for policy in POLICIES
    }
    snapshot_delta: dict[str, dict[str, float]] = {
        policy.name: {} for policy in POLICIES
    }
    anchors_in_probe = {
        anchor_id: row
        for anchor_id, row in anchors.items()
        if any(
            float(start) <= float(row["timestamp_sec"]) <= float(end)
            for start, end in windows
        )
    }

    for window_index, (window_start, window_end) in enumerate(windows, start=1):
        print(
            f"[Tracker policy probe] window {window_index}/{len(windows)} "
            f"{window_start:.3f}-{window_end:.3f}s",
            flush=True,
        )
        _reset_trackers(trackers)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(window_start)) * 1000.0)
        frames_seen = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                source_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if source_sec <= 0.0:
                    source_frame = max(0.0, float(cap.get(cv2.CAP_PROP_POS_FRAMES) or 1.0) - 1.0)
                    source_sec = source_frame / max(1.0, actual_fps)
                if source_sec + 1e-6 < float(window_start):
                    continue
                if source_sec > float(window_end) + 1.0 / max(1.0, actual_fps):
                    break
                frames_seen += 1
                entity_boxes = _entity_boxes(detector, frame)
                policy_rows: dict[str, list[dict[str, Any]]] = {}
                for policy_name, tracker in trackers.items():
                    policy_rows[policy_name] = (
                        tracker.update(entity_boxes, frame)
                        if entity_boxes is not None
                        else []
                    )

                for anchor_id, anchor in anchors_in_probe.items():
                    anchor_ts = float(anchor["timestamp_sec"])
                    delta = abs(source_sec - anchor_ts)
                    if delta > float(args.anchor_time_tolerance_sec):
                        continue
                    for policy_name, rows in policy_rows.items():
                        previous = snapshot_delta[policy_name].get(anchor_id)
                        if previous is not None and delta >= previous:
                            continue
                        snapshot_delta[policy_name][anchor_id] = delta
                        snapshots[policy_name][anchor_id] = {
                            "source_sec": source_sec,
                            "time_delta_sec": delta,
                            "tracked_rows": [dict(row) for row in rows],
                        }
        finally:
            cap.release()
        print(
            f"[Tracker policy probe] window {window_index}: decoded {frames_seen} frames",
            flush=True,
        )

    mapped_by_policy: dict[str, dict[str, dict[str, Any] | None]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        mapped: dict[str, dict[str, Any] | None] = {}
        for anchor_id, anchor in anchors_in_probe.items():
            snapshot = snapshots[policy.name].get(anchor_id)
            if snapshot is None:
                mapped[anchor_id] = None
                continue
            match = _match_anchor(
                [float(v) for v in anchor["box"]],
                snapshot["tracked_rows"],
                min_iou=float(args.min_iou),
                max_center_distance_norm=float(args.max_center_distance_norm),
            )
            if match is not None:
                match["source_sec"] = round(float(snapshot["source_sec"]), 6)
                match["time_delta_sec"] = round(float(snapshot["time_delta_sec"]), 6)
            mapped[anchor_id] = match
        mapped_by_policy[policy.name] = mapped
        metrics[policy.name] = _score_policy(policy.name, anchors, pairs, mapped)

    report = {
        "version": "courtflux_gate_d_v6_tracker_policy_probe_v1",
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "production_policy_selected": False,
        "anti_leakage": {
            "v6_used_for_tracker_diagnosis_not_threshold_tuning": True,
            "policies_are_stock_tracker_type_and_frame_rate_only": True,
            "no_match_threshold_or_buffer_sweep_on_v6": True,
            "human_identity_used_only_to_score_frozen_anchor_relations": True,
            "candidate_box_mapping_uses_geometry_not_identity": True,
            "next_step_if_promising": "develop/validate tracker policy on V1/V3/V4/V5 before any V6 runtime integration",
        },
        "sources": {
            "human_gt": str(Path(args.gt).expanduser().resolve()),
            "anchor_geometry": str(Path(args.geometry).expanduser().resolve()),
            "video": video_path,
            "model": str(model_path),
        },
        "runtime": {
            "device": device,
            "actual_video_fps": actual_fps,
            "detector_conf": float(args.conf),
            "imgsz": int(args.imgsz),
            "anchor_time_tolerance_sec": float(args.anchor_time_tolerance_sec),
            "min_iou": float(args.min_iou),
            "max_center_distance_norm": float(args.max_center_distance_norm),
        },
        "probe_pairs": pairs,
        "windows": [{"start_sec": a, "end_sec": b} for a, b in windows],
        "policy_metadata": policy_metadata,
        "mapped_anchors": mapped_by_policy,
        "metrics": metrics,
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else REPO_ROOT / "evaluation" / "runs" / "p2_v6_gate_d_tracker_policy_probe.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nV6 Gate D tracker-policy probe", flush=True)
    for policy in POLICIES:
        row = metrics[policy.name]
        cov = row["required_anchor_coverage"]
        hand = row["handoff_split_recall"]
        ctrl = row["stable_control_continuity"]
        fm = row["nearby_different_false_merge"]
        print(
            f"{policy.name}: "
            f"anchors={cov['resolved']}/{cov['total']} "
            f"handoff_split={hand['correct']}/{hand['total']} "
            f"control_same={ctrl['correct']}/{ctrl['total']} "
            f"nearby_false_merges={fm['false_merges']}/{fm['resolved']} "
            f"(resolved; total={fm['total']}) "
            f"promising={'YES' if row['promising_for_development_followup'] else 'NO'}",
            flush=True,
        )
    print(f"report: {output_path}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic V6 raw-tracker policy probe. Replays two short windows around frozen "
            "human-confirmed raw-track handoffs and compares stock BoT-SORT/ByteTrack with "
            "legacy 30fps versus actual video fps. All policies share the same YOLO forwards. "
            "No V6 threshold tuning and no production tracker selection are performed."
        )
    )
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--geometry", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--video")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-gap-sec", type=float, default=MAX_LOCAL_GAP_SEC)
    parser.add_argument("--pad-sec", type=float, default=WINDOW_PAD_SEC)
    parser.add_argument("--anchor-time-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--min-iou", type=float, default=0.15)
    parser.add_argument("--max-center-distance-norm", type=float, default=0.10)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
