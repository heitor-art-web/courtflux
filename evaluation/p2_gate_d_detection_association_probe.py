from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_gate_d_tracker_policy_probe as base
from evaluation.p2_gate_d_tracker_policy_probe_v2 import _relation_windows

VERSION = "courtflux_gate_d_v6_detection_association_probe_v1"
MATCH_CONFIG = REPO_ROOT / "evaluation" / "match_configs" / "gate_c_v6.json"
UNMATCHED_COST = 0.95
MAX_MISSED_FRAMES = 18
TEAM_LOCK_MIN_ACCEPTED = 3


@dataclass(frozen=True, slots=True)
class AssocPolicy:
    name: str
    velocity: bool
    team_gate: bool
    torso: bool


POLICIES = (
    AssocPolicy("geometry", velocity=False, team_gate=False, torso=False),
    AssocPolicy("motion", velocity=True, team_gate=False, torso=False),
    AssocPolicy("motion_team", velocity=True, team_gate=True, torso=False),
    AssocPolicy("motion_team_torso", velocity=True, team_gate=True, torso=True),
)


@dataclass(slots=True)
class TrackState:
    track_id: int
    box: list[float]
    center: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float64))
    embedding: np.ndarray | None = None
    team_votes: dict[str, int] = field(default_factory=dict)
    accepted_team_observations: int = 0
    missed: int = 0
    age: int = 1

    @property
    def locked_team(self) -> str | None:
        if self.accepted_team_observations < TEAM_LOCK_MIN_ACCEPTED or not self.team_votes:
            return None
        ordered = sorted(self.team_votes.items(), key=lambda item: item[1], reverse=True)
        if len(ordered) > 1 and ordered[0][1] <= ordered[1][1]:
            return None
        return str(ordered[0][0])


class DetectionAssociator:
    """Probe-only 1:1 player association independent from Ultralytics raw track IDs."""

    def __init__(self, policy: AssocPolicy) -> None:
        self.policy = policy
        self.tracks: dict[int, TrackState] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    @staticmethod
    def _center(box: list[float]) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)

    @staticmethod
    def _diag(box: list[float]) -> float:
        x1, y1, x2, y2 = [float(v) for v in box]
        return max(1.0, math.hypot(x2 - x1, y2 - y1))

    @staticmethod
    def _size_cost(left: list[float], right: list[float]) -> float:
        lx1, ly1, lx2, ly2 = [float(v) for v in left]
        rx1, ry1, rx2, ry2 = [float(v) for v in right]
        la = max(1.0, (lx2 - lx1) * (ly2 - ly1))
        ra = max(1.0, (rx2 - rx1) * (ry2 - ry1))
        return min(2.0, abs(math.log(ra / la))) / 2.0

    @staticmethod
    def _cosine_distance(left: np.ndarray | None, right: np.ndarray | None) -> float:
        if left is None or right is None or left.size == 0 or right.size == 0:
            return 0.5
        ln = float(np.linalg.norm(left))
        rn = float(np.linalg.norm(right))
        if ln <= 1e-9 or rn <= 1e-9:
            return 0.5
        return float(np.clip(1.0 - float(np.dot(left, right) / (ln * rn)), 0.0, 2.0) / 2.0)

    def _predicted_box(self, track: TrackState) -> list[float]:
        if not self.policy.velocity:
            return list(track.box)
        dx, dy = [float(v) for v in track.velocity]
        x1, y1, x2, y2 = track.box
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def _cost(self, track: TrackState, det: dict[str, Any]) -> float:
        det_box = [float(v) for v in det["box"]]
        pred_box = self._predicted_box(track)
        pred_center = self._center(pred_box)
        det_center = self._center(det_box)
        scale = max(self._diag(pred_box), self._diag(det_box), 1.0)
        center_cost = float(np.linalg.norm(det_center - pred_center) / scale)
        iou_cost = 1.0 - base._iou(pred_box, det_box)
        size_cost = self._size_cost(pred_box, det_box)

        # Generic normalized geometry gate, fixed before V6 scoring.
        if center_cost > 1.35 and iou_cost > 0.96:
            return 1e6

        det_team = det.get("team")
        locked_team = track.locked_team
        if self.policy.team_gate and locked_team and det_team and str(det_team) != locked_team:
            return 1e6

        if self.policy.torso:
            torso_cost = self._cosine_distance(track.embedding, det.get("embedding"))
            return 0.48 * center_cost + 0.27 * iou_cost + 0.10 * size_cost + 0.15 * torso_cost
        return 0.57 * center_cost + 0.31 * iou_cost + 0.12 * size_cost

    def _spawn(self, det: dict[str, Any]) -> TrackState:
        box = [float(v) for v in det["box"]]
        embedding = det.get("embedding")
        track = TrackState(
            track_id=self.next_id,
            box=box,
            center=self._center(box),
            embedding=(np.asarray(embedding, dtype=np.float64).copy() if embedding is not None else None),
        )
        self.next_id += 1
        team = det.get("team")
        if team:
            track.team_votes[str(team)] = 1
            track.accepted_team_observations = 1
        self.tracks[track.track_id] = track
        return track

    @staticmethod
    def _hungarian(cost: np.ndarray) -> list[int]:
        """Return assigned column for each row using rectangular Hungarian O(n^3)."""
        n, m = cost.shape
        if n == 0:
            return []
        if n > m:
            raise ValueError("Hungarian implementation expects rows <= columns")
        u = np.zeros(n + 1, dtype=np.float64)
        v = np.zeros(m + 1, dtype=np.float64)
        p = np.zeros(m + 1, dtype=np.int64)
        way = np.zeros(m + 1, dtype=np.int64)
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = np.full(m + 1, np.inf, dtype=np.float64)
            used = np.zeros(m + 1, dtype=bool)
            while True:
                used[j0] = True
                i0 = int(p[j0])
                delta = np.inf
                j1 = 0
                for j in range(1, m + 1):
                    if used[j]:
                        continue
                    cur = float(cost[i0 - 1, j - 1] - u[i0] - v[j])
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = float(minv[j])
                        j1 = j
                if not np.isfinite(delta):
                    break
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = int(way[j0])
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break
        assignment = [-1] * n
        for j in range(1, m + 1):
            if p[j] > 0:
                assignment[int(p[j]) - 1] = j - 1
        return assignment

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active = [track for track in self.tracks.values() if track.missed <= MAX_MISSED_FRAMES]
        if not active:
            spawned = [self._spawn(det) for det in detections]
            return [self._row(track, det) for track, det in zip(spawned, detections)]

        n_tracks = len(active)
        n_det = len(detections)
        cost = np.full((n_tracks, n_det + n_tracks), UNMATCHED_COST, dtype=np.float64)
        if n_det:
            for i, track in enumerate(active):
                for j, det in enumerate(detections):
                    cost[i, j] = self._cost(track, det)
            # Each track gets its own unmatched dummy; other dummy columns remain expensive.
            for i in range(n_tracks):
                for j in range(n_tracks):
                    cost[i, n_det + j] = UNMATCHED_COST if i == j else 1e6

        assignment = self._hungarian(cost)
        matched_det: set[int] = set()
        rows: list[dict[str, Any]] = []
        for i, track in enumerate(active):
            col = assignment[i]
            if col < 0 or col >= n_det or float(cost[i, col]) >= UNMATCHED_COST:
                track.missed += 1
                track.age += 1
                continue
            det = detections[col]
            matched_det.add(col)
            old_center = track.center.copy()
            new_box = [float(v) for v in det["box"]]
            new_center = self._center(new_box)
            displacement = new_center - old_center
            track.velocity = 0.65 * track.velocity + 0.35 * displacement
            track.center = new_center
            track.box = new_box
            track.missed = 0
            track.age += 1
            team = det.get("team")
            if team:
                track.team_votes[str(team)] = track.team_votes.get(str(team), 0) + 1
                track.accepted_team_observations += 1
            emb = det.get("embedding")
            if emb is not None:
                arr = np.asarray(emb, dtype=np.float64)
                track.embedding = arr.copy() if track.embedding is None else 0.90 * track.embedding + 0.10 * arr
            rows.append(self._row(track, det))

        for j, det in enumerate(detections):
            if j in matched_det:
                continue
            track = self._spawn(det)
            rows.append(self._row(track, det))

        expired = [tid for tid, track in self.tracks.items() if track.missed > MAX_MISSED_FRAMES]
        for tid in expired:
            self.tracks.pop(tid, None)
        return rows

    @staticmethod
    def _row(track: TrackState, det: dict[str, Any]) -> dict[str, Any]:
        return {
            "track_id": int(track.track_id),
            "box": [float(v) for v in track.box],
            "confidence": float(det.get("confidence", 0.0)),
            "team": det.get("team"),
            "locked_team": track.locked_team,
        }


def _is_player_name(name: str) -> bool:
    value = str(name).lower().replace("-", "_").replace(" ", "_")
    if "referee" in value or value in {"ref", "official", "officials"}:
        return False
    return "player" in value or "person" in value


def _decode_detections(model: Any, analyzer: Any, frame: np.ndarray, *, conf: float, imgsz: int, device: str) -> list[dict[str, Any]]:
    results = model.predict(frame, conf=float(conf), iou=0.5, imgsz=int(imgsz), max_det=100, device=device, verbose=False)
    result = results[0] if results else None
    boxes = getattr(result, "boxes", None) if result is not None else None
    if boxes is None or len(boxes) == 0:
        return []
    cpu = boxes.cpu()
    xyxy = cpu.xyxy.numpy()
    confs = cpu.conf.numpy()
    classes = cpu.cls.numpy()
    names = getattr(model, "names", {}) or {}
    output: list[dict[str, Any]] = []
    h, w = frame.shape[:2]
    for coords, score, cls_value in zip(xyxy, confs, classes):
        cls_id = int(cls_value)
        name = str(names.get(cls_id, cls_id) if isinstance(names, dict) else names[cls_id]).lower()
        if not _is_player_name(name) or float(score) < float(conf):
            continue
        x1, y1, x2, y2 = [float(v) for v in coords[:4]]
        if x2 <= x1 or y2 <= y1:
            continue
        ix1, iy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
        ix2, iy2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
        crop = frame[iy1:iy2, ix1:ix2]
        evidence = analyzer.score_teams(crop) if crop.size else {}
        embedding = evidence.get("color_embedding") or []
        output.append({
            "box": [x1, y1, x2, y2],
            "confidence": float(score),
            "team": evidence.get("best_team") if evidence.get("evidence_accepted") else None,
            "team_best_score": float(evidence.get("best_score", 0.0) or 0.0),
            "team_margin": float(evidence.get("margin", 0.0) or 0.0),
            "embedding": np.asarray(embedding, dtype=np.float64) if embedding else None,
        })
    return output


def _scoped_pairs(pairs: dict[str, list[dict[str, Any]]], mapped: dict[str, dict[str, Any] | None]) -> dict[str, list[dict[str, Any]]]:
    active = set(mapped)
    return {
        key: [dict(row) for row in rows if str(row.get("left")) in active and str(row.get("right")) in active]
        for key, rows in pairs.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt = base._load_json(Path(args.gt).expanduser().resolve())
    geometry = base._load_json(Path(args.geometry).expanduser().resolve())
    v6 = base._load_json(base.DEFAULT_V6)
    anchors = base._anchors_by_id(geometry)
    base._validate_geometry_against_gt(gt, anchors)
    pairs = base._derive_probe_pairs(anchors, max_gap_sec=float(args.max_gap_sec))
    windows = _relation_windows(anchors, pairs, pad_sec=float(args.pad_sec))

    video_path = base._resolve_video(v6, args.video)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    from services.analytics.analyzer import resolve_device
    from services.vision.team_only_analyzer import TeamOnlyAnalyzer
    from ultralytics import YOLO

    device = resolve_device(args.device)
    model = YOLO(str(model_path))
    model.to(device)
    team_analyzer = TeamOnlyAnalyzer(profiling_path=str(MATCH_CONFIG))
    associators = {policy.name: DetectionAssociator(policy) for policy in POLICIES}

    snapshots: dict[str, dict[str, dict[str, Any]]] = {policy.name: {} for policy in POLICIES}
    snapshot_delta: dict[str, dict[str, float]] = {policy.name: {} for policy in POLICIES}
    anchors_in_probe = {
        anchor_id: row for anchor_id, row in anchors.items()
        if any(float(a) <= float(row["timestamp_sec"]) <= float(b) for a, b in windows)
    }

    print(
        "[Detection association probe] "
        f"windows={[(round(a,3), round(b,3)) for a,b in windows]} "
        f"anchors={len(anchors_in_probe)} policies={[p.name for p in POLICIES]}",
        flush=True,
    )

    for window_index, (window_start, window_end) in enumerate(windows, start=1):
        for associator in associators.values():
            associator.reset()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, window_start) * 1000.0)
        frames_seen = 0
        print(f"[Detection association probe] window {window_index}/{len(windows)} {window_start:.3f}-{window_end:.3f}s", flush=True)
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                source_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                if source_sec <= 0.0:
                    source_frame = max(0.0, float(cap.get(cv2.CAP_PROP_POS_FRAMES) or 1.0) - 1.0)
                    source_sec = source_frame / max(1.0, fps)
                if source_sec + 1e-6 < window_start:
                    continue
                if source_sec > window_end + 1.0 / max(1.0, fps):
                    break
                frames_seen += 1
                detections = _decode_detections(model, team_analyzer, frame, conf=args.conf, imgsz=args.imgsz, device=device)
                policy_rows = {name: tracker.update(detections) for name, tracker in associators.items()}
                for anchor_id, anchor in anchors_in_probe.items():
                    delta = abs(source_sec - float(anchor["timestamp_sec"]))
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
                            "rows": [dict(row) for row in rows],
                        }
        finally:
            cap.release()
        print(f"[Detection association probe] window {window_index}: decoded {frames_seen} frames", flush=True)

    mapped_by_policy: dict[str, dict[str, dict[str, Any] | None]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        mapped: dict[str, dict[str, Any] | None] = {}
        for anchor_id, anchor in anchors_in_probe.items():
            snapshot = snapshots[policy.name].get(anchor_id)
            if snapshot is None:
                mapped[anchor_id] = None
                continue
            match = base._match_anchor(
                [float(v) for v in anchor["box"]], snapshot["rows"],
                min_iou=float(args.min_iou), max_center_distance_norm=float(args.max_center_distance_norm),
            )
            if match is not None:
                match["source_sec"] = round(float(snapshot["source_sec"]), 6)
                match["time_delta_sec"] = round(float(snapshot["time_delta_sec"]), 6)
            mapped[anchor_id] = match
        mapped_by_policy[policy.name] = mapped
        scoped = _scoped_pairs(pairs, mapped)
        metrics[policy.name] = base._score_policy(policy.name, anchors, scoped, mapped)
        metrics[policy.name]["scope_integrity"] = {
            "only_relations_inside_replayed_windows": True,
            "scored_pair_counts": {key: len(rows) for key, rows in scoped.items()},
        }

    report = {
        "version": VERSION,
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "runtime_code_modified": False,
        "anti_leakage": {
            "no_ultralytics_track_id_used": True,
            "human_identity_used_only_for_frozen_relation_scoring": True,
            "team_evidence_uses_existing_frozen_gate_c_v6_colour_config": True,
            "no_parameter_or_threshold_sweep_on_v6": True,
            "policies_are_fixed_ablation_families": True,
            "next_step_if_promising": "validate unchanged association policy on V1/V3/V4/V5 before runtime integration",
        },
        "sources": {
            "human_gt": str(Path(args.gt).expanduser().resolve()),
            "anchor_geometry": str(Path(args.geometry).expanduser().resolve()),
            "video": video_path,
            "model": str(model_path),
            "team_config": str(MATCH_CONFIG),
        },
        "windows": [{"start_sec": a, "end_sec": b} for a, b in windows],
        "probe_pairs": pairs,
        "mapped_anchors": mapped_by_policy,
        "metrics": metrics,
        "policies": [policy.__dict__ if hasattr(policy, "__dict__") else {
            "name": policy.name, "velocity": policy.velocity, "team_gate": policy.team_gate, "torso": policy.torso
        } for policy in POLICIES],
    }

    output = Path(args.output).expanduser().resolve() if args.output else REPO_ROOT / "evaluation" / "runs" / "p2_v6_gate_d_detection_association_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nV6 Gate D detection-level association probe", flush=True)
    for policy in POLICIES:
        row = metrics[policy.name]
        cov = row["required_anchor_coverage"]
        hand = row["handoff_split_recall"]
        ctrl = row["stable_control_continuity"]
        fm = row["nearby_different_false_merge"]
        same = row["nearby_same_raw_continuity"]
        print(
            f"{policy.name}: anchors={cov['resolved']}/{cov['total']} "
            f"handoff_split={hand['correct']}/{hand['total']} "
            f"control_same={ctrl['correct']}/{ctrl['total']} "
            f"nearby_same={same['correct']}/{same['total']} "
            f"nearby_false_merges={fm['false_merges']}/{fm['resolved']} "
            f"promising={'YES' if row['promising_for_development_followup'] else 'NO'}",
            flush=True,
        )
    print(f"report: {output}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostic V6 detection-level player association probe. No runtime integration or V6 threshold tuning.")
    parser.add_argument("--gt", default=str(base.DEFAULT_GT))
    parser.add_argument("--geometry", default=str(base.DEFAULT_GEOMETRY))
    parser.add_argument("--video")
    parser.add_argument("--model", default=str(base.DEFAULT_MODEL))
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-gap-sec", type=float, default=base.MAX_LOCAL_GAP_SEC)
    parser.add_argument("--pad-sec", type=float, default=base.WINDOW_PAD_SEC)
    parser.add_argument("--anchor-time-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--min-iou", type=float, default=0.15)
    parser.add_argument("--max-center-distance-norm", type=float, default=0.10)
    parser.add_argument("--output")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
