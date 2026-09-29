from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import p2_v6_gate_d_reid_preflight as base
from evaluation import p2_v6_gate_d_tracklet_signature_preflight as sig
from evaluation.p2_v6_gate_d_reid_preflight_rawtrack import _raw_track_prediction


DEFAULT_LOCAL_WINDOW_FRAMES = 42
DEFAULT_SAMPLES = 8
SAME_TARGET = 0.85
FALSE_MERGE_MAX = 0.05


@dataclass(frozen=True)
class Node:
    observation_id: str
    track_id: int
    frame_id: int
    timestamp_sec: float
    team_id: str | None


class UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self.parent = {key: key for key in keys}
        self.members = {key: {key} for key in keys}

    def find(self, key: str) -> str:
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, left: str, right: str) -> str:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return a
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a].update(self.members[b])
        del self.members[b]
        return a

    def cluster(self, key: str) -> set[str]:
        return self.members[self.find(key)]



def _team_vote(predictions: list[dict[str, Any]], track_id: int) -> tuple[str | None, float]:
    weights: dict[str, float] = defaultdict(float)
    total = 0.0
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        try:
            candidate = int(prediction.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        if candidate != int(track_id):
            continue
        team_id = prediction.get("team_id")
        if team_id is None:
            continue
        confidence = float(prediction.get("team_confidence") or prediction.get("confidence") or 0.0)
        confidence = max(0.05, min(1.0, confidence))
        weights[str(team_id)] += confidence
        total += confidence
    if not weights or total <= 0.0:
        return None, 0.0
    team_id, score = max(weights.items(), key=lambda item: item[1])
    return team_id, float(score / total)



def _collect_local_candidates(
    states_path: Path,
    observations: dict[str, dict[str, Any]],
    *,
    window_frames: int,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    set[tuple[str, str]],
    dict[str, tuple[str | None, float]],
]:
    result = {key: [] for key in observations}
    team_votes: dict[str, list[dict[str, Any]]] = {key: [] for key in observations}
    cannot_link: set[tuple[str, str]] = set()
    targets = list(observations.values())

    # observation-specific windows avoid treating raw track IDs as identities.
    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            try:
                frame_id = int(row.get("frame_id", -10**9))
            except (TypeError, ValueError):
                continue
            timestamp = row.get("timestamp_sec")
            if timestamp is None:
                continue
            predictions = [p for p in (row.get("team_predictions") or []) if isinstance(p, dict)]

            active_obs: list[str] = []
            for observation in targets:
                observation_id = str(observation["id"])
                target_frame = int(observation["target_frame_id"])
                if abs(frame_id - target_frame) > int(window_frames):
                    continue
                prediction = _raw_track_prediction(row, int(observation["track_id"]))
                if prediction is None:
                    continue
                box = prediction.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                box_f = [float(value) for value in box]
                result[observation_id].append(
                    {
                        "frame_id": frame_id,
                        "timestamp_sec": float(timestamp),
                        "box": box_f,
                        "quality": sig._bbox_quality(box_f, prediction),
                    }
                )
                team_votes[observation_id].extend(predictions)
                active_obs.append(observation_id)

            # If two different raw tracks are visible in the same frame they cannot be one person.
            for i, left_id in enumerate(active_obs):
                left_track = int(observations[left_id]["track_id"])
                for right_id in active_obs[i + 1 :]:
                    right_track = int(observations[right_id]["track_id"])
                    if left_track == right_track:
                        continue
                    cannot_link.add(tuple(sorted((left_id, right_id))))

    teams: dict[str, tuple[str | None, float]] = {}
    for observation_id, rows in team_votes.items():
        teams[observation_id] = _team_vote(rows, int(observations[observation_id]["track_id"]))
    return result, cannot_link, teams



def _selected_samples(
    candidates: dict[str, list[dict[str, Any]]],
    observations: dict[str, dict[str, Any]],
    *,
    samples: int,
    window_frames: int,
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for observation_id, observation in observations.items():
        rows = sig._select_samples(
            candidates.get(observation_id, []),
            target_frame=int(observation["target_frame_id"]),
            sample_count=int(samples),
            window_frames=int(window_frames),
        )
        selected[observation_id] = rows
        if not rows:
            missing.append(observation_id)
    if missing:
        raise RuntimeError("No raw-track samples for: " + ", ".join(missing))
    return selected



def _build_descriptors(
    video_path: Path,
    selected: dict[str, list[dict[str, Any]]],
) -> dict[str, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    descriptors: dict[str, list[np.ndarray]] = {key: [] for key in selected}
    try:
        for observation_id, rows in selected.items():
            for row in rows:
                crop = sig._decode_torso(
                    capture,
                    timestamp_sec=float(row["timestamp_sec"]),
                    box=list(row["box"]),
                )
                descriptors[observation_id].append(sig._descriptor(crop))
    finally:
        capture.release()
    return descriptors



def _node_similarity(
    left: Node,
    right: Node,
    descriptors: dict[str, list[np.ndarray]],
    cannot_link: set[tuple[str, str]],
) -> tuple[float, dict[str, Any]]:
    pair_key = tuple(sorted((left.observation_id, right.observation_id)))
    if pair_key in cannot_link:
        return -1.0, {"cannot_link": True, "reason": "co_visible"}

    visual = float(
        sig._tracklet_similarity(
            descriptors[left.observation_id],
            descriptors[right.observation_id],
        )
    )
    gap_sec = abs(float(left.timestamp_sec) - float(right.timestamp_sec))
    same_raw_track = int(left.track_id) == int(right.track_id)

    # Team disagreement is a strong negative, but not absolute because Gate C can flicker.
    team_bonus = 0.0
    if left.team_id is not None and right.team_id is not None:
        team_bonus = 0.06 if left.team_id == right.team_id else -0.22

    # Raw-track continuity is useful only locally and only if local torso evidence agrees.
    track_bonus = 0.0
    if same_raw_track:
        if gap_sec <= 10.0 and visual >= 0.58:
            track_bonus = 0.10
        elif visual < 0.46:
            track_bonus = -0.12

    # Short temporal gaps get a small prior; long gaps must be carried by appearance/global consistency.
    temporal_bonus = 0.0
    if gap_sec <= 3.0:
        temporal_bonus = 0.035
    elif gap_sec <= 8.0:
        temporal_bonus = 0.018

    score = visual + team_bonus + track_bonus + temporal_bonus
    return float(score), {
        "cannot_link": False,
        "visual": round(visual, 6),
        "team_bonus": round(team_bonus, 6),
        "track_bonus": round(track_bonus, 6),
        "temporal_bonus": round(temporal_bonus, 6),
        "gap_sec": round(gap_sec, 3),
        "same_raw_track": bool(same_raw_track),
    }



def _cluster_conflicts(
    left_members: set[str],
    right_members: set[str],
    cannot_link: set[tuple[str, str]],
) -> bool:
    for left in left_members:
        for right in right_members:
            if tuple(sorted((left, right))) in cannot_link:
                return True
    return False



def _cluster_at_threshold(
    nodes: dict[str, Node],
    pair_scores: dict[tuple[str, str], tuple[float, dict[str, Any]]],
    cannot_link: set[tuple[str, str]],
    threshold: float,
) -> dict[str, int]:
    uf = UnionFind(list(nodes))
    ranked = sorted(
        (
            (score, left, right)
            for (left, right), (score, _) in pair_scores.items()
            if score >= float(threshold)
        ),
        reverse=True,
    )
    for score, left, right in ranked:
        left_root = uf.find(left)
        right_root = uf.find(right)
        if left_root == right_root:
            continue
        left_members = set(uf.members[left_root])
        right_members = set(uf.members[right_root])
        if _cluster_conflicts(left_members, right_members, cannot_link):
            continue
        uf.union(left, right)

    root_to_index: dict[str, int] = {}
    labels: dict[str, int] = {}
    for observation_id in nodes:
        root = uf.find(observation_id)
        if root not in root_to_index:
            root_to_index[root] = len(root_to_index)
        labels[observation_id] = root_to_index[root]
    return labels



def _score_labels(
    gt: dict[str, Any],
    labels: dict[str, int],
) -> dict[str, Any]:
    same_pairs = [p for p in (gt.get("same_pairs") or []) if isinstance(p, dict)]
    different_pairs = [p for p in (gt.get("different_pairs") or []) if isinstance(p, dict)]
    same_correct = sum(
        labels.get(str(pair.get("left"))) == labels.get(str(pair.get("right")))
        for pair in same_pairs
    )
    false_merges = sum(
        labels.get(str(pair.get("left"))) == labels.get(str(pair.get("right")))
        for pair in different_pairs
    )
    same_recall = same_correct / len(same_pairs) if same_pairs else 0.0
    false_merge_rate = false_merges / len(different_pairs) if different_pairs else 0.0
    return {
        "same_correct": int(same_correct),
        "same_total": len(same_pairs),
        "same_recall": float(same_recall),
        "different_false_merges": int(false_merges),
        "different_total": len(different_pairs),
        "different_false_merge_rate": float(false_merge_rate),
        "gate_feasible": bool(
            same_recall >= SAME_TARGET and false_merge_rate <= FALSE_MERGE_MAX
        ),
    }



def run(args: argparse.Namespace) -> dict[str, Any]:
    gt_path = Path(args.gt).expanduser().resolve()
    states_path = Path(args.states).expanduser().resolve() if args.states else base._latest_states_file()
    video_path = Path(args.video).expanduser().resolve()
    gt = base._load_json(gt_path)
    observations = base._mapped_observations(gt)

    print("[Global association] 1/5 scanning local raw-track evidence", flush=True)
    candidates, cannot_link, team_votes = _collect_local_candidates(
        states_path,
        observations,
        window_frames=int(args.window_frames),
    )
    selected = _selected_samples(
        candidates,
        observations,
        samples=int(args.samples),
        window_frames=int(args.window_frames),
    )

    print("[Global association] 2/5 building local torso tracklet signatures", flush=True)
    descriptors = _build_descriptors(video_path, selected)

    nodes: dict[str, Node] = {}
    for observation_id, observation in observations.items():
        team_id, _ = team_votes.get(observation_id, (None, 0.0))
        nodes[observation_id] = Node(
            observation_id=observation_id,
            track_id=int(observation["track_id"]),
            frame_id=int(observation["target_frame_id"]),
            timestamp_sec=float(observation.get("human_source_sec") or 0.0),
            team_id=team_id,
        )

    print("[Global association] 3/5 building constrained identity graph", flush=True)
    pair_scores: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
    observation_ids = sorted(nodes)
    for i, left_id in enumerate(observation_ids):
        for right_id in observation_ids[i + 1 :]:
            pair_scores[(left_id, right_id)] = _node_similarity(
                nodes[left_id],
                nodes[right_id],
                descriptors,
                cannot_link,
            )

    # Oracle sweep is diagnostic only: asks whether the graph representation contains enough signal
    # before any runtime integration. It must never become the production threshold directly.
    finite_scores = sorted(
        set(score for score, _ in pair_scores.values() if score > -0.999),
        reverse=True,
    )
    thresholds: list[float] = []
    if finite_scores:
        thresholds.append(finite_scores[0] + 1e-6)
        thresholds.extend((a + b) / 2.0 for a, b in zip(finite_scores, finite_scores[1:]))
        thresholds.append(finite_scores[-1] - 1e-6)
    else:
        thresholds = [1.0]

    print("[Global association] 4/5 sweeping graph operating point", flush=True)
    trials: list[dict[str, Any]] = []
    best_labels: dict[str, int] = {}
    for threshold in thresholds:
        labels = _cluster_at_threshold(nodes, pair_scores, cannot_link, threshold)
        metrics = _score_labels(gt, labels)
        trial = {
            "threshold": float(threshold),
            "cluster_count": len(set(labels.values())),
            **metrics,
        }
        trials.append(trial)

    feasible = [row for row in trials if row["gate_feasible"]]
    pool = feasible or [
        row
        for row in trials
        if row["different_false_merge_rate"] <= FALSE_MERGE_MAX + 1e-12
    ] or trials
    best = max(
        pool,
        key=lambda row: (
            row["same_recall"],
            -row["different_false_merge_rate"],
            -row["cluster_count"],
        ),
    )
    best_labels = _cluster_at_threshold(
        nodes,
        pair_scores,
        cannot_link,
        float(best["threshold"]),
    )

    print("[Global association] 5/5 writing report", flush=True)
    report = {
        "version": "courtflux_gate_d_v6_global_association_preflight_v1",
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "anti_leakage": {
            "human_player_label_used_for_clustering": False,
            "game_player_id_used": False,
            "human_pair_labels_used_only_for_oracle_evaluation": True,
            "production_threshold_must_not_be_copied_from_oracle": True,
        },
        "sources": {
            "ground_truth": str(gt_path),
            "states": str(states_path),
            "video": str(video_path),
        },
        "policy": {
            "local_window_frames": int(args.window_frames),
            "samples_per_observation": int(args.samples),
            "co_visibility_hard_cannot_link": True,
            "same_raw_track_not_a_hard_identity": True,
            "team_from_runtime_predictions": True,
            "semantic_jersey_reading": False,
        },
        "best_oracle_operating_point": best,
        "signal_feasible_for_gate_d": bool(best["gate_feasible"]),
        "labels_at_best_point": best_labels,
        "team_votes": {
            key: {"team_id": value[0], "confidence": round(float(value[1]), 6)}
            for key, value in team_votes.items()
        },
        "cannot_link_count": len(cannot_link),
        "pair_scores": [
            {
                "left": left,
                "right": right,
                "score": round(float(score), 6),
                **diag,
            }
            for (left, right), (score, diag) in sorted(pair_scores.items())
        ],
        "trial_count": len(trials),
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else states_path.parent / "p2_v6_gate_d_global_association_preflight.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("\nV6 Gate D global-association preflight", flush=True)
    print(
        f"SAME: {best['same_correct']}/{best['same_total']} = "
        f"{100.0 * float(best['same_recall']):.2f}%",
        flush=True,
    )
    print(
        f"DIFFERENT false merges: {best['different_false_merges']}/{best['different_total']} = "
        f"{100.0 * float(best['different_false_merge_rate']):.2f}%",
        flush=True,
    )
    print(f"anonymous identity clusters: {best['cluster_count']}", flush=True)
    print(f"oracle graph threshold: {float(best['threshold']):.6f}", flush=True)
    print(
        "signal feasible for Gate D: " + ("YES" if best["gate_feasible"] else "NO"),
        flush=True,
    )
    print(f"report: {output_path}", flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V6 Gate-D offline global association diagnostic. Clusters frozen endpoint "
            "observations using runtime team evidence, local torso signatures, raw-track "
            "continuity as a soft prior, and co-visibility as a hard cannot-link."
        )
    )
    parser.add_argument("--gt", default=str(base.DEFAULT_GT))
    parser.add_argument("--states", default=None)
    parser.add_argument("--video", default=str(base.DEFAULT_VIDEO))
    parser.add_argument("--output", default=None)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_LOCAL_WINDOW_FRAMES)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.window_frames < 1:
        raise SystemExit("--window-frames must be >= 1")
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
