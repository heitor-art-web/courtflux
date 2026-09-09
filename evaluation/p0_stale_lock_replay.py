from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import load_jsonl
from evaluation.p0_quality_gate import GateThresholds, build_summary, evaluate_pair
from services.vision.ball_tracker import BallAssociationDecision, TemporalBallTracker


@dataclass
class ShadowChallenger:
    center: np.ndarray | None = None
    area: float | None = None
    frame_id: int | None = None
    streak: int = 0

    def reset(self) -> None:
        self.center = None
        self.area = None
        self.frame_id = None
        self.streak = 0


class SafeStaleLockReplayTracker:
    """Experimental replay-only stale-lock escape.

    This policy does not promote an outside-gate challenger. It only invalidates an
    active track after a stronger outside-gate hypothesis remains spatially consistent
    for several frames. The invalidation frame emits no location. On later frames the
    frozen tracker bootstrap logic is allowed to initialize again.

    It is intentionally evaluation-only until frozen V1-V5 evidence shows that the
    precision/coverage tradeoff is safe.
    """

    def __init__(
        self,
        *,
        min_detector_confidence: float = 0.45,
        detector_confidence_margin: float = 0.15,
        confirm_frames: int = 3,
        radius_norm: float = 0.04,
        min_size_similarity: float = 0.25,
        **tracker_kwargs: Any,
    ) -> None:
        self.tracker = TemporalBallTracker(reacquire_enabled=False, **tracker_kwargs)
        self.min_detector_confidence = float(min_detector_confidence)
        self.detector_confidence_margin = float(detector_confidence_margin)
        self.confirm_frames = max(2, int(confirm_frames))
        self.radius_norm = max(0.005, float(radius_norm))
        self.min_size_similarity = max(0.0, min(float(min_size_similarity), 1.0))
        self.shadow = ShadowChallenger()
        self.invalidations = 0

    @staticmethod
    def _area(candidate: dict[str, Any]) -> float:
        box = candidate.get("box") or [0, 0, 0, 0]
        return max(float(box[2]) - float(box[0]), 1.0) * max(
            float(box[3]) - float(box[1]), 1.0
        )

    @staticmethod
    def _area_similarity(a: float | None, b: float | None) -> float:
        if a is None or b is None or a <= 0.0 or b <= 0.0:
            return 0.5
        ratio = max(float(a) / float(b), 1e-6)
        return float(math.exp(-abs(math.log(ratio))))

    @staticmethod
    def _detector_conf(candidate: dict[str, Any]) -> float:
        return float(candidate.get("detector_confidence", candidate.get("confidence", 0.0)))

    def _clear_active(self) -> None:
        tracker = self.tracker
        tracker.last_center = None
        tracker.last_box = None
        tracker.last_frame_id = None
        tracker.last_area = None
        tracker.last_confidence = None
        tracker.velocity[:] = 0.0
        tracker.track_state = "LOST"
        tracker._reset_challenger()

    def _update_shadow(
        self,
        candidate: dict[str, Any] | None,
        *,
        frame_id: int,
        diagonal: float,
    ) -> int:
        if candidate is None:
            self.shadow.reset()
            return 0

        center = np.asarray(candidate["center"], dtype=np.float64)
        area = self._area(candidate)
        consistent = False
        if self.shadow.center is not None and self.shadow.frame_id is not None:
            dt = int(frame_id) - int(self.shadow.frame_id)
            if 0 < dt <= 2:
                distance_norm = float(np.linalg.norm(center - self.shadow.center) / diagonal)
                allowed = self.radius_norm * float(max(dt, 1))
                consistent = (
                    distance_norm <= allowed
                    and self._area_similarity(area, self.shadow.area) >= self.min_size_similarity
                )

        self.shadow.streak = self.shadow.streak + 1 if consistent else 1
        self.shadow.center = center
        self.shadow.area = area
        self.shadow.frame_id = int(frame_id)
        return self.shadow.streak

    def select(
        self,
        candidates: list[dict[str, Any]],
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> BallAssociationDecision:
        candidates = list(candidates or [])
        active = self.tracker.last_center is not None and self.tracker.last_frame_id is not None
        if not active or not candidates:
            self.shadow.reset()
            return self.tracker.select(candidates, frame_id, frame_shape)

        pre_scores = [
            self.tracker.inspect_candidate(candidate, frame_id, frame_shape)
            for candidate in candidates
        ]
        plausible = [
            (idx, candidate, score)
            for idx, (candidate, score) in enumerate(zip(candidates, pre_scores))
            if score is not None and bool(score.plausible)
        ]
        selected_plausible = (
            max(plausible, key=lambda item: float(item[2].score)) if plausible else None
        )
        selected_detector_conf = (
            self._detector_conf(selected_plausible[1]) if selected_plausible else 0.0
        )

        outside = [
            (idx, candidate, score)
            for idx, (candidate, score) in enumerate(zip(candidates, pre_scores))
            if score is not None and not bool(score.plausible)
        ]
        minimum = max(
            self.min_detector_confidence,
            selected_detector_conf + self.detector_confidence_margin,
        )
        eligible = [
            item for item in outside if self._detector_conf(item[1]) >= minimum
        ]
        challenger = (
            max(eligible, key=lambda item: self._detector_conf(item[1]))[1]
            if eligible
            else None
        )

        height, width = frame_shape[:2]
        diagonal = max(math.hypot(float(width), float(height)), 1.0)
        streak = self._update_shadow(challenger, frame_id=frame_id, diagonal=diagonal)
        if streak >= self.confirm_frames:
            self.invalidations += 1
            self._clear_active()
            self.shadow.reset()
            return BallAssociationDecision(
                ball=None,
                source="stale_track_invalidated",
                selected_index=None,
                score=None,
                distance_norm=None,
                gate_norm=None,
                rejected_candidates=len(candidates),
                hold_age_frames=0,
                track_state="LOST",
                reacquired=False,
            )

        return self.tracker.select(candidates, frame_id, frame_shape)


def parse_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    run_id, gt_id = [part.strip() for part in value.split("=", 1)]
    if not run_id or not gt_id:
        raise argparse.ArgumentTypeError("Expected RUN_ID=GT_RUN_ID")
    return run_id, gt_id


def candidate_from_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    box = trace.get("box")
    center = trace.get("center")
    if not isinstance(box, list) or len(box) != 4:
        return None
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        center = [
            (float(box[0]) + float(box[2])) / 2.0,
            (float(box[1]) + float(box[3])) / 2.0,
        ]

    association_conf = trace.get("association_confidence")
    if association_conf is None:
        association_conf = trace.get("rank_score")
    if association_conf is None:
        association_conf = trace.get("confidence", 0.0)
    detector_conf = trace.get("detector_confidence", trace.get("confidence", association_conf))

    return {
        "box": [int(round(float(value))) for value in box],
        "center": (float(center[0]), float(center[1])),
        "confidence": float(association_conf or 0.0),
        "detector_confidence": float(detector_conf or 0.0),
        "interpolated": False,
        "candidate_source": trace.get("source"),
    }


def candidates_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    association = row.get("ball_association") or {}
    traces = association.get("candidates") or []
    candidates: list[dict[str, Any]] = []
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        candidate = candidate_from_trace(trace)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def resolve_frame_shape(run_dir: Path, fallback: tuple[int, int] | None) -> tuple[int, int, int]:
    if fallback is not None:
        return int(fallback[1]), int(fallback[0]), 3

    candidates: list[str] = []
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("video_source", "video_path"):
                value = config.get(key)
                if value:
                    candidates.append(str(value))
        except Exception:
            pass
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            extra = summary.get("extra") or {}
            for value in (extra.get("video_path"), summary.get("video_source")):
                if value:
                    candidates.append(str(value))
        except Exception:
            pass

    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            continue
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        if width > 0 and height > 0:
            return height, width, 3

    raise RuntimeError(
        f"Could not resolve frame dimensions for {run_dir}. "
        "Pass --frame-width and --frame-height."
    )


def decision_row(original: dict[str, Any], decision: BallAssociationDecision) -> dict[str, Any]:
    row = dict(original)
    original_association = original.get("ball_association") or {}
    association = dict(original_association)
    association.update(
        {
            "source": decision.source,
            "selected_index": decision.selected_index,
            "score": decision.score,
            "distance_norm": decision.distance_norm,
            "gate_norm": decision.gate_norm,
            "rejected_candidates": decision.rejected_candidates,
            "hold_age_frames": decision.hold_age_frames,
            "track_state": decision.track_state,
            "reacquired": decision.reacquired,
            "trajectory_center": (
                None
                if decision.ball is None
                else [float(decision.ball["center"][0]), float(decision.ball["center"][1])]
            ),
        }
    )
    row["ball_association"] = association
    direct = decision.ball is not None and not bool(decision.ball.get("interpolated", False))
    row["selected_ball"] = bool(direct)
    row["selected_ball_center"] = (
        [float(decision.ball["center"][0]), float(decision.ball["center"][1])]
        if direct and decision.ball is not None
        else None
    )
    row["ball_source"] = decision.source
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def replay_run(
    run_dir: Path,
    *,
    frame_shape: tuple[int, int, int],
    safe: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows = load_jsonl(run_dir / "frames.jsonl")
    tracker: Any
    if safe:
        tracker = SafeStaleLockReplayTracker()
    else:
        tracker = TemporalBallTracker(reacquire_enabled=False)

    output: list[dict[str, Any]] = []
    for row in rows:
        if not bool(row.get("valid_gameplay")):
            output.append(dict(row))
            continue
        frame_id = int(row.get("frame_id", 0))
        candidates = candidates_from_row(row)
        decision = tracker.select(candidates, frame_id, frame_shape)
        output.append(decision_row(row, decision))

    invalidations = int(getattr(tracker, "invalidations", 0)) if safe else 0
    return output, invalidations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen Smart candidate traces through the baseline tracker and one "
            "conservative stale-lock invalidation policy. No detector inference is run."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--output", default="evaluation/p0_stale_lock_replay")
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    args = parser.parse_args()

    fallback = None
    if args.frame_width is not None or args.frame_height is not None:
        if not args.frame_width or not args.frame_height:
            raise SystemExit("Pass both --frame-width and --frame-height")
        fallback = (int(args.frame_width), int(args.frame_height))

    run_root = (ROOT / args.run_root).resolve()
    gt_root = (ROOT / args.gt_root).resolve()
    output_root = (ROOT / args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_results: list[dict[str, Any]] = []
    safe_results: list[dict[str, Any]] = []
    invalidations_total = 0

    for run_id, gt_id in args.pair:
        run_dir = run_root / run_id
        frame_shape = resolve_frame_shape(run_dir, fallback)
        baseline_rows, _ = replay_run(run_dir, frame_shape=frame_shape, safe=False)
        safe_rows, invalidations = replay_run(run_dir, frame_shape=frame_shape, safe=True)
        invalidations_total += invalidations

        baseline_path = output_root / f"{run_id}_baseline.jsonl"
        safe_path = output_root / f"{run_id}_safe_stale_lock.jsonl"
        write_jsonl(baseline_path, baseline_rows)
        write_jsonl(safe_path, safe_rows)

        baseline = evaluate_pair(
            baseline_path,
            gt_root / gt_id / "annotations.jsonl",
            radius_px=40.0,
            catastrophic_radius_px=160.0,
            timestamp_tolerance_sec=0.001,
        )
        safe_result = evaluate_pair(
            safe_path,
            gt_root / gt_id / "annotations.jsonl",
            radius_px=40.0,
            catastrophic_radius_px=160.0,
            timestamp_tolerance_sec=0.001,
        )
        baseline["run_id"] = run_id
        baseline["gt_run_id"] = gt_id
        safe_result["run_id"] = run_id
        safe_result["gt_run_id"] = gt_id
        safe_result["stale_track_invalidations"] = invalidations
        baseline_results.append(baseline)
        safe_results.append(safe_result)

    thresholds = GateThresholds()
    baseline_summary = build_summary(
        baseline_results, thresholds=thresholds, catastrophic_radius_px=160.0
    )
    safe_summary = build_summary(
        safe_results, thresholds=thresholds, catastrophic_radius_px=160.0
    )

    recorded_path = ROOT / "evaluation/p0_quality_gate_smart_v1/summary.json"
    recorded = None
    if recorded_path.exists():
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))

    payload = {
        "scope": "P0 offline association stale-lock replay",
        "policy": {
            "name": "safe_stale_lock_invalidation_v1",
            "promotion": False,
            "reacquisition": False,
            "min_detector_confidence": 0.45,
            "detector_confidence_margin": 0.15,
            "confirm_frames": 3,
            "radius_norm": 0.04,
            "min_size_similarity": 0.25,
            "note": (
                "Single predeclared policy using existing recovery evidence thresholds; "
                "not a threshold sweep."
            ),
        },
        "invalidations_total": invalidations_total,
        "recorded_reference": (
            None if recorded is None else recorded.get("aggregate")
        ),
        "baseline_replay": baseline_summary,
        "safe_stale_lock_replay": safe_summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def show(label: str, summary: dict[str, Any]) -> None:
        aggregate = summary["aggregate"]
        association = aggregate["p0_2_association"]
        trajectory = aggregate["p0_3_trajectory"]
        state = aggregate["p0_4_state_robustness"]
        system = aggregate["p0_5_system"]
        print(f"\n{label}")
        print("-" * 72)
        print(f"direct coverage:        {association['direct_correct_coverage_micro_pct']:7.2f}%")
        print(f"direct precision:       {association['direct_output_precision_micro_pct']:7.2f}%")
        print(f"association miss:       {association['association_opportunity_miss_rate_micro_pct']:7.2f}%")
        print(f"usable coverage:        {trajectory['usable_correct_visible_coverage_micro_pct']:7.2f}%")
        print(f"predicted precision:    {trajectory['predicted_output_precision_micro_pct']:7.2f}%")
        print(f"not-visible false:      {state['not_visible_false_state_rate_micro_pct']:7.2f}%")
        print(f"catastrophic events:    {state['catastrophic_error_events']:7d}")
        print(f"unsupported gap P90:    {system['unsupported_gap_p90_frames']!s:>7} frames")

    print("P0 STALE-LOCK OFFLINE REPLAY")
    print("=" * 72)
    show("BASELINE REPLAY", baseline_summary)
    show("SAFE STALE-LOCK INVALIDATION", safe_summary)
    print(f"\nstale-track invalidations: {invalidations_total}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
