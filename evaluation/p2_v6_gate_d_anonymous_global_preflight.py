from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERSION = "courtflux_gate_d_v6_anonymous_global_preflight_v1"
DEFAULT_GT = REPO_ROOT / "evaluation" / "ground_truth" / "gate_d_v6_human_ball_carrier_gt_v1.json"

MAX_CONTIG_GAP_FRAMES = 3
MAX_TRACKLET_FRAMES = 120
CROSSING_DISTANCE_PLAYER_HEIGHTS = 0.85
CROSSING_IOU_MIN = 0.05
KINEMATIC_JUMP_PLAYER_HEIGHTS = 1.25
STRONG_TEAM_POSTERIOR = 0.90
DEFAULT_SLOTS_PER_TEAM = 6
DEFAULT_BEAM_WIDTH = 512
SKIP_TRACKLET_COST = 2.20

SAME_RECALL_TARGET = 0.85
DIFFERENT_FALSE_MERGE_MAX = 0.05


@dataclass(frozen=True, slots=True)
class Observation:
    frame_id: int
    timestamp_sec: float
    raw_track_id: int
    box: tuple[float, float, float, float]
    team: str | None
    team_confidence: float
    active_on_court: bool | None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def width(self) -> float:
        return max(1.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(1.0, self.box[3] - self.box[1])

    @property
    def aspect(self) -> float:
        return self.width / self.height


@dataclass(frozen=True, slots=True)
class MicroTracklet:
    index: int
    raw_track_id: int
    observations: tuple[Observation, ...]
    start_reason: str
    best_team: str | None
    team_posterior: float

    @property
    def frame_start(self) -> int:
        return self.observations[0].frame_id

    @property
    def frame_end(self) -> int:
        return self.observations[-1].frame_id

    @property
    def timestamp_start(self) -> float:
        return self.observations[0].timestamp_sec

    @property
    def timestamp_end(self) -> float:
        return self.observations[-1].timestamp_sec

    @property
    def start_center(self) -> tuple[float, float]:
        return _mean_center(self.observations[: min(5, len(self.observations))])

    @property
    def end_center(self) -> tuple[float, float]:
        return _mean_center(self.observations[max(0, len(self.observations) - 5) :])

    @property
    def mean_height(self) -> float:
        return sum(row.height for row in self.observations) / len(self.observations)

    @property
    def mean_aspect(self) -> float:
        return sum(row.aspect for row in self.observations) / len(self.observations)

    @property
    def start_velocity(self) -> tuple[float, float]:
        return _velocity(self.observations[: min(6, len(self.observations))])

    @property
    def end_velocity(self) -> tuple[float, float]:
        return _velocity(self.observations[max(0, len(self.observations) - 6) :])


@dataclass(slots=True)
class BeamState:
    cost: float
    last_by_slot: tuple[int, ...]
    parent: "BeamState | None"
    tracklet_index: int
    assigned_slot: int


def _latest_states_file() -> Path:
    candidates = sorted(
        path
        for path in (REPO_ROOT / "evaluation" / "runs").glob("*/p2_states.jsonl")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            "No evaluation/runs/*/p2_states.jsonl found. Pass --states explicitly."
        )
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _valid_box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    la = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    ra = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    return inter / max(1e-9, la + ra - inter)


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _mean_center(rows: Iterable[Observation]) -> tuple[float, float]:
    values = list(rows)
    if not values:
        return (0.0, 0.0)
    xs = [row.center[0] for row in values]
    ys = [row.center[1] for row in values]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _velocity(rows: Iterable[Observation]) -> tuple[float, float]:
    values = list(rows)
    if len(values) < 2:
        return (0.0, 0.0)
    first, last = values[0], values[-1]
    frames = max(1, last.frame_id - first.frame_id)
    return (
        (last.center[0] - first.center[0]) / frames,
        (last.center[1] - first.center[1]) / frames,
    )


def _collect_observations(states_path: Path) -> tuple[dict[int, list[Observation]], dict[int, list[Observation]]]:
    by_track: dict[int, list[Observation]] = {}
    by_frame: dict[int, list[Observation]] = {}
    with states_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number}: {exc}") from exc
            if not isinstance(row, dict) or row.get("gameplay_present") is not True:
                continue
            try:
                frame_id = int(row.get("frame_id"))
                timestamp_sec = float(row.get("timestamp_sec"))
            except (TypeError, ValueError):
                continue
            predictions = row.get("team_predictions") or []
            for prediction in predictions:
                if not isinstance(prediction, dict):
                    continue
                if prediction.get("identity_eligible") is False:
                    continue
                if prediction.get("active_on_court") is False:
                    continue
                try:
                    raw_track_id = int(prediction.get("track_id", -1))
                except (TypeError, ValueError):
                    continue
                if raw_track_id < 0:
                    continue
                box = _valid_box(prediction.get("box"))
                if box is None:
                    continue
                team = prediction.get("team")
                team_text = str(team) if team is not None else None
                try:
                    team_conf = float(
                        prediction.get("team_confidence", prediction.get("confidence", 0.0))
                        or 0.0
                    )
                except (TypeError, ValueError):
                    team_conf = 0.0
                active = prediction.get("active_on_court")
                active_value = active if isinstance(active, bool) else None
                obs = Observation(
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    raw_track_id=raw_track_id,
                    box=box,
                    team=team_text,
                    team_confidence=max(0.0, min(1.0, team_conf)),
                    active_on_court=active_value,
                )
                by_track.setdefault(raw_track_id, []).append(obs)
                by_frame.setdefault(frame_id, []).append(obs)
    for rows in by_track.values():
        rows.sort(key=lambda item: item.frame_id)
    return by_track, by_frame


def _crossing_flags(by_frame: dict[int, list[Observation]]) -> set[tuple[int, int]]:
    flagged: set[tuple[int, int]] = set()
    for frame_id, rows in by_frame.items():
        for i, left in enumerate(rows):
            if left.team is None or left.team_confidence < 0.60:
                continue
            for right in rows[i + 1 :]:
                if right.team != left.team or right.team_confidence < 0.60:
                    continue
                scale = max(1.0, (left.height + right.height) * 0.5)
                normalized = _distance(left.center, right.center) / scale
                close = normalized <= CROSSING_DISTANCE_PLAYER_HEIGHTS
                overlap = _iou(left.box, right.box) >= CROSSING_IOU_MIN
                if close or overlap:
                    flagged.add((left.raw_track_id, frame_id))
                    flagged.add((right.raw_track_id, frame_id))
    return flagged


def _dominant_team(rows: list[Observation]) -> tuple[str | None, float]:
    weights: dict[str, float] = {}
    for row in rows:
        if row.team is None:
            continue
        weights[row.team] = weights.get(row.team, 0.0) + max(0.05, row.team_confidence)
    if not weights:
        return None, 0.0
    ordered = sorted(weights.items(), key=lambda item: (-item[1], item[0]))
    total = sum(value for _, value in ordered)
    return ordered[0][0], ordered[0][1] / max(total, 1e-9)


def _make_tracklet(index: int, raw_track_id: int, rows: list[Observation], reason: str) -> MicroTracklet:
    best_team, posterior = _dominant_team(rows)
    return MicroTracklet(
        index=index,
        raw_track_id=raw_track_id,
        observations=tuple(rows),
        start_reason=reason,
        best_team=best_team,
        team_posterior=posterior,
    )


def _build_micro_tracklets(
    by_track: dict[int, list[Observation]],
    crossing_flags: set[tuple[int, int]],
) -> list[MicroTracklet]:
    output: list[MicroTracklet] = []
    next_index = 0
    for raw_track_id in sorted(by_track):
        rows = by_track[raw_track_id]
        if not rows:
            continue
        current: list[Observation] = [rows[0]]
        start_reason = "raw_track_start"
        previous_crossing = (raw_track_id, rows[0].frame_id) in crossing_flags
        for row in rows[1:]:
            prev = current[-1]
            gap = row.frame_id - prev.frame_id
            crossing = (raw_track_id, row.frame_id) in crossing_flags
            normalized_jump = _distance(prev.center, row.center) / max(
                1.0, (prev.height + row.height) * 0.5
            )
            reason: str | None = None
            if gap > MAX_CONTIG_GAP_FRAMES:
                reason = "raw_gap"
            elif row.frame_id - current[0].frame_id >= MAX_TRACKLET_FRAMES:
                reason = "periodic"
            elif crossing != previous_crossing:
                reason = "crossing_enter" if crossing else "crossing_exit"
            elif normalized_jump > KINEMATIC_JUMP_PLAYER_HEIGHTS:
                reason = "kinematic_jump"

            if reason is not None:
                output.append(_make_tracklet(next_index, raw_track_id, current, start_reason))
                next_index += 1
                current = [row]
                start_reason = reason
            else:
                current.append(row)
            previous_crossing = crossing
        if current:
            output.append(_make_tracklet(next_index, raw_track_id, current, start_reason))
            next_index += 1

    output.sort(key=lambda item: (item.frame_start, item.frame_end, item.raw_track_id, item.index))
    # Re-index after chronological sorting so beam back-references are compact/stable.
    return [
        MicroTracklet(
            index=index,
            raw_track_id=item.raw_track_id,
            observations=item.observations,
            start_reason=item.start_reason,
            best_team=item.best_team,
            team_posterior=item.team_posterior,
        )
        for index, item in enumerate(output)
    ]


def _transition_cost(previous: MicroTracklet, current: MicroTracklet) -> float:
    if previous.frame_end >= current.frame_start:
        return math.inf
    gap = max(1, current.frame_start - previous.frame_end)
    scale = max(1.0, (previous.mean_height + current.mean_height) * 0.5)

    # Motion matters only locally. Across a long camera absence, image coordinates are not
    # identity evidence, so the global search keeps several slot hypotheses alive instead
    # of pretending a long-range geometric extrapolation is meaningful.
    if gap <= 30:
        horizon = float(gap)
        center_weight = 0.52
        velocity_weight = 0.18
    elif gap <= 120:
        horizon = 30.0
        center_weight = 0.30
        velocity_weight = 0.10
    elif gap <= 600:
        horizon = 12.0
        center_weight = 0.10
        velocity_weight = 0.04
    else:
        horizon = 0.0
        center_weight = 0.0
        velocity_weight = 0.0

    predicted = (
        previous.end_center[0] + previous.end_velocity[0] * horizon,
        previous.end_center[1] + previous.end_velocity[1] * horizon,
    )
    center_cost = min(3.0, _distance(predicted, current.start_center) / scale) / 3.0
    velocity_delta = _distance(previous.end_velocity, current.start_velocity)
    velocity_cost = min(2.0, velocity_delta / max(0.5, scale / 30.0)) / 2.0
    aspect_cost = min(
        1.0,
        abs(math.log(max(1e-6, current.mean_aspect / max(previous.mean_aspect, 1e-6)))) / 0.8,
    )
    height_cost = min(
        1.0,
        abs(math.log(max(1e-6, current.mean_height / max(previous.mean_height, 1e-6)))) / 1.2,
    )
    gap_cost = min(1.0, gap / 600.0)

    cost = (
        center_weight * center_cost
        + velocity_weight * velocity_cost
        + 0.10 * aspect_cost
        + 0.08 * height_cost
        + 0.24 * gap_cost
    )

    if previous.raw_track_id == current.raw_track_id:
        if current.start_reason in {"crossing_enter", "crossing_exit", "kinematic_jump"}:
            raw_bonus = 0.0
        elif current.start_reason == "periodic":
            raw_bonus = -0.42
        else:
            raw_bonus = -0.18
        cost += raw_bonus

    return max(0.0, cost)


def _slot_layout(team_names: list[str], slots_per_team: int) -> tuple[list[str], list[int]]:
    slot_teams: list[str] = []
    local_indices: list[int] = []
    for team in team_names:
        for local in range(int(slots_per_team)):
            slot_teams.append(team)
            local_indices.append(local)
    return slot_teams, local_indices


def _candidate_slots(
    tracklet: MicroTracklet,
    slot_teams: list[str],
) -> list[int]:
    if tracklet.best_team is None or tracklet.team_posterior < STRONG_TEAM_POSTERIOR:
        return list(range(len(slot_teams)))
    return [index for index, team in enumerate(slot_teams) if team == tracklet.best_team]


def _team_penalty(tracklet: MicroTracklet, slot_team: str) -> float:
    if tracklet.best_team is None or tracklet.best_team == slot_team:
        return 0.0
    return 0.55 * max(0.25, tracklet.team_posterior)


def _assign_global(
    tracklets: list[MicroTracklet],
    *,
    team_names: list[str],
    slots_per_team: int,
    beam_width: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    slot_teams, local_indices = _slot_layout(team_names, slots_per_team)
    root = BeamState(
        cost=0.0,
        last_by_slot=tuple([-1] * len(slot_teams)),
        parent=None,
        tracklet_index=-1,
        assigned_slot=-1,
    )
    beam = [root]

    for index, tracklet in enumerate(tracklets):
        candidates: dict[tuple[int, ...], BeamState] = {}
        for state in beam:
            for slot in _candidate_slots(tracklet, slot_teams):
                previous_index = state.last_by_slot[slot]
                if previous_index >= 0:
                    previous = tracklets[previous_index]
                    if previous.frame_end >= tracklet.frame_start:
                        continue
                    transition = _transition_cost(previous, tracklet)
                    if not math.isfinite(transition):
                        continue
                    assignment_cost = transition
                else:
                    # Prefer the five normal on-court slots; a sixth slot is available as
                    # conservative overflow for substitutions/false-positive clutter.
                    assignment_cost = 0.86 + (0.42 if local_indices[slot] >= 5 else 0.0)

                assignment_cost += _team_penalty(tracklet, slot_teams[slot])
                assignment_cost += slot * 1e-7  # deterministic tie-break only
                new_last = list(state.last_by_slot)
                new_last[slot] = index
                new_last_tuple = tuple(new_last)
                child = BeamState(
                    cost=state.cost + assignment_cost,
                    last_by_slot=new_last_tuple,
                    parent=state,
                    tracklet_index=index,
                    assigned_slot=slot,
                )
                previous_best = candidates.get(new_last_tuple)
                if previous_best is None or child.cost < previous_best.cost:
                    candidates[new_last_tuple] = child

            # Explicit abstention path. It is deliberately expensive but prevents one
            # dubious/extra detection from corrupting every persistent identity slot.
            skipped = BeamState(
                cost=state.cost + SKIP_TRACKLET_COST,
                last_by_slot=state.last_by_slot,
                parent=state,
                tracklet_index=index,
                assigned_slot=-1,
            )
            key = state.last_by_slot
            previous_best = candidates.get(key)
            if previous_best is None or skipped.cost < previous_best.cost:
                candidates[key] = skipped

        if not candidates:
            raise RuntimeError(f"Global association exhausted all hypotheses at tracklet {index}")
        beam = sorted(candidates.values(), key=lambda item: item.cost)[: max(1, int(beam_width))]

    best = min(beam, key=lambda item: item.cost)
    assignments: dict[int, int] = {}
    cursor: BeamState | None = best
    while cursor is not None and cursor.tracklet_index >= 0:
        assignments[cursor.tracklet_index] = cursor.assigned_slot
        cursor = cursor.parent

    return assignments, {
        "objective_cost": round(best.cost, 6),
        "beam_width": int(beam_width),
        "slots_per_team": int(slots_per_team),
        "slot_teams": slot_teams,
        "final_hypotheses": len(beam),
        "skipped_tracklets": sum(1 for value in assignments.values() if value < 0),
    }


def _map_gt_observations(
    gt: dict[str, Any],
    tracklets: list[MicroTracklet],
    assignments: dict[int, int],
) -> dict[str, dict[str, Any]]:
    by_raw: dict[int, list[MicroTracklet]] = {}
    for tracklet in tracklets:
        by_raw.setdefault(tracklet.raw_track_id, []).append(tracklet)

    mapped: dict[str, dict[str, Any]] = {}
    for row in gt.get("mapped_observations") or []:
        if not isinstance(row, dict):
            continue
        observation_id = str(row.get("id") or "")
        endpoint = row.get("endpoint") or {}
        if not observation_id or not isinstance(endpoint, dict):
            continue
        try:
            frame_id = int(endpoint.get("frame_id"))
            raw_track_id = int(endpoint.get("track_id"))
        except (TypeError, ValueError):
            continue
        candidates = by_raw.get(raw_track_id, [])
        containing = [
            item for item in candidates if item.frame_start <= frame_id <= item.frame_end
        ]
        if containing:
            tracklet = min(
                containing,
                key=lambda item: min(abs(obs.frame_id - frame_id) for obs in item.observations),
            )
            nearest = min(abs(obs.frame_id - frame_id) for obs in tracklet.observations)
        else:
            nearby = [
                item
                for item in candidates
                if min(abs(obs.frame_id - frame_id) for obs in item.observations) <= 3
            ]
            if not nearby:
                mapped[observation_id] = {
                    "resolved": False,
                    "reason": "raw_track_not_present_near_endpoint",
                    "frame_id": frame_id,
                    "raw_track_id": raw_track_id,
                }
                continue
            tracklet = min(
                nearby,
                key=lambda item: min(abs(obs.frame_id - frame_id) for obs in item.observations),
            )
            nearest = min(abs(obs.frame_id - frame_id) for obs in tracklet.observations)
        slot = assignments.get(tracklet.index, -1)
        mapped[observation_id] = {
            "resolved": slot >= 0,
            "reason": "assigned" if slot >= 0 else "global_abstention",
            "frame_id": frame_id,
            "raw_track_id": raw_track_id,
            "tracklet_index": tracklet.index,
            "tracklet_start_reason": tracklet.start_reason,
            "nearest_frame_delta": int(nearest),
            "anonymous_slot": None if slot < 0 else int(slot),
            "best_team_evidence": tracklet.best_team,
            "team_posterior": round(tracklet.team_posterior, 6),
        }
    return mapped


def _score_pairs(gt: dict[str, Any], mapped: dict[str, dict[str, Any]]) -> dict[str, Any]:
    same_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []

    for pair in gt.get("same_pairs") or []:
        left_id, right_id = str(pair.get("left")), str(pair.get("right"))
        left, right = mapped.get(left_id, {}), mapped.get(right_id, {})
        resolved = bool(left.get("resolved")) and bool(right.get("resolved"))
        correct = bool(
            resolved and left.get("anonymous_slot") == right.get("anonymous_slot")
        )
        same_rows.append(
            {
                "id": pair.get("id"),
                "left": left_id,
                "right": right_id,
                "resolved": resolved,
                "correct": correct,
                "left_slot": left.get("anonymous_slot"),
                "right_slot": right.get("anonymous_slot"),
            }
        )

    for pair in gt.get("different_pairs") or []:
        left_id, right_id = str(pair.get("left")), str(pair.get("right"))
        left, right = mapped.get(left_id, {}), mapped.get(right_id, {})
        resolved = bool(left.get("resolved")) and bool(right.get("resolved"))
        false_merge = bool(
            resolved and left.get("anonymous_slot") == right.get("anonymous_slot")
        )
        diff_rows.append(
            {
                "id": pair.get("id"),
                "left": left_id,
                "right": right_id,
                "resolved": resolved,
                "false_merge": false_merge,
                "left_slot": left.get("anonymous_slot"),
                "right_slot": right.get("anonymous_slot"),
            }
        )

    same_correct = sum(bool(row["correct"]) for row in same_rows)
    false_merges = sum(bool(row["false_merge"]) for row in diff_rows)
    same_total = len(same_rows)
    diff_total = len(diff_rows)
    same_recall = same_correct / same_total if same_total else 0.0
    false_merge_rate = false_merges / diff_total if diff_total else 0.0
    return {
        "same": {
            "correct": same_correct,
            "total": same_total,
            "recall": same_recall,
            "pairs": same_rows,
        },
        "different": {
            "false_merges": false_merges,
            "total": diff_total,
            "false_merge_rate": false_merge_rate,
            "pairs": diff_rows,
        },
        "signal_feasible_for_gate_d": bool(
            same_recall >= SAME_RECALL_TARGET
            and false_merge_rate <= DIFFERENT_FALSE_MERGE_MAX
        ),
    }


def _simultaneous_conflicts(
    tracklets: list[MicroTracklet], assignments: dict[int, int]
) -> list[dict[str, Any]]:
    by_slot: dict[int, list[MicroTracklet]] = {}
    for tracklet in tracklets:
        slot = assignments.get(tracklet.index, -1)
        if slot >= 0:
            by_slot.setdefault(slot, []).append(tracklet)
    conflicts: list[dict[str, Any]] = []
    for slot, rows in by_slot.items():
        ordered = sorted(rows, key=lambda item: (item.frame_start, item.frame_end))
        for left, right in zip(ordered, ordered[1:]):
            if left.frame_end >= right.frame_start:
                conflicts.append(
                    {
                        "slot": slot,
                        "left_tracklet": left.index,
                        "right_tracklet": right.index,
                        "overlap": [right.frame_start, min(left.frame_end, right.frame_end)],
                    }
                )
    return conflicts


def _raw_track_baseline(gt: dict[str, Any]) -> dict[str, Any]:
    obs = {
        str(row.get("id")): int((row.get("endpoint") or {}).get("track_id", -1))
        for row in gt.get("mapped_observations") or []
        if isinstance(row, dict) and isinstance(row.get("endpoint"), dict)
    }
    same = gt.get("same_pairs") or []
    diff = gt.get("different_pairs") or []
    same_correct = sum(obs.get(str(row.get("left"))) == obs.get(str(row.get("right"))) for row in same)
    false_merges = sum(obs.get(str(row.get("left"))) == obs.get(str(row.get("right"))) for row in diff)
    return {
        "same_correct": int(same_correct),
        "same_total": len(same),
        "same_recall": (same_correct / len(same) if same else 0.0),
        "different_false_merges": int(false_merges),
        "different_total": len(diff),
        "different_false_merge_rate": (false_merges / len(diff) if diff else 0.0),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gt_path = Path(args.gt).expanduser().resolve()
    states_path = (
        Path(args.states).expanduser().resolve() if args.states else _latest_states_file()
    )
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    if not states_path.exists():
        raise FileNotFoundError(states_path)

    gt = _load_json(gt_path)
    print("[Anonymous global preflight] 1/5 reading team predictions", flush=True)
    by_track, by_frame = _collect_observations(states_path)
    print(
        f"[Anonymous global preflight] raw tracks={len(by_track)} frames={len(by_frame)}",
        flush=True,
    )

    print("[Anonymous global preflight] 2/5 detecting crossings + micro-tracklets", flush=True)
    crossing_flags = _crossing_flags(by_frame)
    tracklets = _build_micro_tracklets(by_track, crossing_flags)
    crossing_tracklets = sum(
        item.start_reason in {"crossing_enter", "crossing_exit"} for item in tracklets
    )

    team_counts: dict[str, int] = {}
    for tracklet in tracklets:
        if tracklet.best_team:
            team_counts[tracklet.best_team] = team_counts.get(tracklet.best_team, 0) + 1
    team_names = [name for name, _ in sorted(team_counts.items(), key=lambda item: (-item[1], item[0]))[:2]]
    if len(team_names) < 2:
        raise RuntimeError(f"Expected two team evidence buckets, found {team_names}")

    print(
        f"[Anonymous global preflight] tracklets={len(tracklets)} crossing-boundaries={crossing_tracklets} teams={team_names}",
        flush=True,
    )
    print("[Anonymous global preflight] 3/5 global beam association", flush=True)
    assignments, solver = _assign_global(
        tracklets,
        team_names=team_names,
        slots_per_team=int(args.slots_per_team),
        beam_width=int(args.beam_width),
    )

    print("[Anonymous global preflight] 4/5 mapping frozen human anchors", flush=True)
    mapped = _map_gt_observations(gt, tracklets, assignments)
    score = _score_pairs(gt, mapped)
    conflicts = _simultaneous_conflicts(tracklets, assignments)
    score["simultaneous_conflicts"] = len(conflicts)
    if conflicts:
        score["signal_feasible_for_gate_d"] = False

    output = {
        "version": VERSION,
        "diagnostic_only": True,
        "formal_gate_d_score": False,
        "purpose": (
            "Test whether anonymous game-local identity can be recovered globally from existing "
            "P1/P2 geometry, timing, co-visibility and soft team evidence without jersey/P5 evidence."
        ),
        "boundary": {
            "jersey_number_used": False,
            "jersey_ocr_used": False,
            "roster_used": False,
            "jersey_trained_features_used": False,
            "game_player_id_used_as_input": False,
            "human_identity_used_by_solver": False,
            "human_gt_use": "score_only_after_global_assignment",
        },
        "sources": {
            "ground_truth": str(gt_path),
            "states": str(states_path),
            "observation_source": "team_predictions only",
        },
        "micro_tracklets": {
            "count": len(tracklets),
            "raw_tracks": len(by_track),
            "crossing_flags": len(crossing_flags),
            "crossing_boundary_tracklets": crossing_tracklets,
            "max_tracklet_frames": MAX_TRACKLET_FRAMES,
            "max_contiguous_gap_frames": MAX_CONTIG_GAP_FRAMES,
            "kinematic_jump_player_heights": KINEMATIC_JUMP_PLAYER_HEIGHTS,
            "crossing_distance_player_heights": CROSSING_DISTANCE_PLAYER_HEIGHTS,
        },
        "solver": solver,
        "raw_track_baseline": _raw_track_baseline(gt),
        "mapped_observations": mapped,
        "score": score,
        "simultaneous_conflict_rows": conflicts,
        "tracklet_assignments": [
            {
                "tracklet_index": item.index,
                "raw_track_id": item.raw_track_id,
                "frame_start": item.frame_start,
                "frame_end": item.frame_end,
                "start_reason": item.start_reason,
                "best_team": item.best_team,
                "team_posterior": round(item.team_posterior, 6),
                "anonymous_slot": (
                    None if assignments.get(item.index, -1) < 0 else int(assignments[item.index])
                ),
            }
            for item in tracklets
        ],
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else states_path.parent / "p2_v6_gate_d_anonymous_global_preflight.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("[Anonymous global preflight] 5/5 report", flush=True)
    same = score["same"]
    different = score["different"]
    resolved_count = sum(bool(row.get("resolved")) for row in mapped.values())
    print("\nV6 Gate D anonymous global-association preflight", flush=True)
    print(f"mapped anchors: {resolved_count}/{len(mapped)}", flush=True)
    print(
        f"SAME: {same['correct']}/{same['total']} = {100.0 * float(same['recall']):.2f}%",
        flush=True,
    )
    print(
        "DIFFERENT false merges: "
        f"{different['false_merges']}/{different['total']} = "
        f"{100.0 * float(different['false_merge_rate']):.2f}%",
        flush=True,
    )
    print(f"simultaneous conflicts: {len(conflicts)}", flush=True)
    print(
        "signal feasible for Gate D: "
        + ("YES" if score["signal_feasible_for_gate_d"] else "NO"),
        flush=True,
    )
    print(f"report: {output_path}", flush=True)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline anonymous Gate-D global association preflight. Reads existing p2_states; "
            "does not run YOLO, OCR, jersey models or the Gate-D resolver."
        )
    )
    parser.add_argument("--gt", default=str(DEFAULT_GT))
    parser.add_argument("--states", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--slots-per-team", type=int, default=DEFAULT_SLOTS_PER_TEAM)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.slots_per_team < 5:
        raise SystemExit("--slots-per-team must be >= 5")
    if args.beam_width < 1:
        raise SystemExit("--beam-width must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
