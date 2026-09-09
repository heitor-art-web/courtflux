from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.ball_error_review import load_jsonl
from evaluation.p0_quality_gate import GateThresholds, build_summary, evaluate_pair
from evaluation.p0_stale_lock_replay import (
    candidates_from_row,
    decision_row,
    parse_pair,
    resolve_frame_shape,
    write_jsonl,
)
from services.vision.ball_tracker import BallAssociationDecision, TemporalBallTracker


@dataclass
class Hypothesis:
    center: np.ndarray
    box: list[int]
    area: float
    velocity: np.ndarray
    last_frame_id: int
    streak: int
    misses: int
    score: float
    detector_confidence: float
    candidate: dict[str, Any]


class MultiHypothesisTracker:
    """Replay-only conservative multi-hypothesis ball association.

    Unlike the production single-track gate, every detector candidate may seed an
    independent temporal hypothesis. Competing hypotheses are retained in a small beam,
    so a wrong active path cannot make all outside-gate evidence disappear.

    The policy deliberately abstains unless the best direct hypothesis has temporal
    confirmation, sufficient quality, and a margin over the next confirmed hypothesis.
    It does not emit predicted holds and it never performs hard reacquisition. The goal
    of this experiment is to test whether multi-hypothesis association can improve P0.2
    and P0.4 reliability before any production change is considered.
    """

    def __init__(
        self,
        *,
        beam_width: int = 5,
        confirm_frames: int = 3,
        min_emit_score: float = 0.55,
        emit_margin: float = 0.15,
        max_internal_misses: int = 2,
        base_gate_norm: float = 0.12,
        gate_growth_per_frame: float = 0.025,
        max_gate_norm: float = 0.28,
        velocity_alpha: float = 0.40,
    ) -> None:
        self.beam_width = max(2, int(beam_width))
        self.confirm_frames = max(2, int(confirm_frames))
        self.min_emit_score = float(min_emit_score)
        self.emit_margin = max(0.0, float(emit_margin))
        self.max_internal_misses = max(0, int(max_internal_misses))
        self.base_gate_norm = max(0.02, float(base_gate_norm))
        self.gate_growth_per_frame = max(0.0, float(gate_growth_per_frame))
        self.max_gate_norm = max(self.base_gate_norm, float(max_gate_norm))
        self.velocity_alpha = max(0.0, min(float(velocity_alpha), 1.0))
        self.hypotheses: list[Hypothesis] = []
        self.abstentions = 0
        self.ambiguous_abstentions = 0

    @staticmethod
    def _area(candidate: dict[str, Any]) -> float:
        box = candidate.get("box") or [0, 0, 0, 0]
        return max(float(box[2]) - float(box[0]), 1.0) * max(
            float(box[3]) - float(box[1]), 1.0
        )

    @staticmethod
    def _area_similarity(a: float, b: float) -> float:
        if a <= 0.0 or b <= 0.0:
            return 0.5
        ratio = max(float(a) / float(b), 1e-6)
        return float(math.exp(-abs(math.log(ratio))))

    @staticmethod
    def _association_conf(candidate: dict[str, Any]) -> float:
        return max(0.0, min(float(candidate.get("confidence", 0.0)), 1.0))

    @staticmethod
    def _detector_conf(candidate: dict[str, Any]) -> float:
        return max(
            0.0,
            min(
                float(candidate.get("detector_confidence", candidate.get("confidence", 0.0))),
                1.0,
            ),
        )

    def _gate(self, dt: int) -> float:
        return min(
            self.base_gate_norm + self.gate_growth_per_frame * max(int(dt) - 1, 0),
            self.max_gate_norm,
        )

    def _spawn(self, candidate: dict[str, Any], frame_id: int) -> Hypothesis:
        center = np.asarray(candidate["center"], dtype=np.float64)
        confidence = self._association_conf(candidate)
        return Hypothesis(
            center=center,
            box=[int(v) for v in candidate["box"]],
            area=self._area(candidate),
            velocity=np.zeros(2, dtype=np.float64),
            last_frame_id=int(frame_id),
            streak=1,
            misses=0,
            score=confidence,
            detector_confidence=self._detector_conf(candidate),
            candidate=dict(candidate),
        )

    def _extend(
        self,
        hypothesis: Hypothesis,
        candidate: dict[str, Any],
        *,
        frame_id: int,
        diagonal: float,
    ) -> Hypothesis | None:
        dt = int(frame_id) - int(hypothesis.last_frame_id)
        if dt <= 0 or dt > self.max_internal_misses + 1:
            return None

        center = np.asarray(candidate["center"], dtype=np.float64)
        predicted = hypothesis.center + hypothesis.velocity * float(dt)
        distance_norm = float(np.linalg.norm(center - predicted) / diagonal)
        gate = self._gate(dt)
        if distance_norm > gate:
            return None

        area = self._area(candidate)
        proximity = max(0.0, 1.0 - distance_norm / max(gate, 1e-6))
        size_similarity = self._area_similarity(area, hypothesis.area)
        confidence = self._association_conf(candidate)
        observation_score = 0.58 * confidence + 0.32 * proximity + 0.10 * size_similarity

        # Finite-memory average: evidence from roughly the last five observations matters,
        # preventing an early false lock from accumulating unbounded inertia.
        memory = min(max(hypothesis.streak, 1), 5)
        score = (hypothesis.score * float(memory) + observation_score) / float(memory + 1)

        measured_velocity = (center - hypothesis.center) / float(dt)
        velocity = (
            (1.0 - self.velocity_alpha) * hypothesis.velocity
            + self.velocity_alpha * measured_velocity
        )
        return Hypothesis(
            center=center,
            box=[int(v) for v in candidate["box"]],
            area=area,
            velocity=velocity,
            last_frame_id=int(frame_id),
            streak=hypothesis.streak + 1,
            misses=0,
            score=float(score),
            detector_confidence=self._detector_conf(candidate),
            candidate=dict(candidate),
        )

    def _carry(self, hypothesis: Hypothesis, frame_id: int) -> Hypothesis | None:
        age = int(frame_id) - int(hypothesis.last_frame_id)
        if age <= 0 or age > self.max_internal_misses:
            return None
        return Hypothesis(
            center=hypothesis.center.copy(),
            box=list(hypothesis.box),
            area=float(hypothesis.area),
            velocity=hypothesis.velocity.copy(),
            last_frame_id=int(hypothesis.last_frame_id),
            streak=int(hypothesis.streak),
            misses=age,
            score=float(hypothesis.score * (0.90 ** age)),
            detector_confidence=float(hypothesis.detector_confidence),
            candidate=dict(hypothesis.candidate),
        )

    @staticmethod
    def _rank_key(hypothesis: Hypothesis) -> tuple[float, int, float]:
        return (
            float(hypothesis.score),
            int(hypothesis.streak),
            float(hypothesis.detector_confidence),
        )

    def _dedupe(self, hypotheses: list[Hypothesis], diagonal: float) -> list[Hypothesis]:
        kept: list[Hypothesis] = []
        radius = 0.04 * diagonal
        for hypothesis in sorted(hypotheses, key=self._rank_key, reverse=True):
            duplicate = any(
                np.linalg.norm(hypothesis.center - existing.center) <= radius
                and abs(hypothesis.last_frame_id - existing.last_frame_id) <= 1
                for existing in kept
            )
            if not duplicate:
                kept.append(hypothesis)
            if len(kept) >= self.beam_width:
                break
        return kept

    def select(
        self,
        candidates: list[dict[str, Any]],
        frame_id: int,
        frame_shape: tuple[int, ...],
    ) -> BallAssociationDecision:
        candidates = list(candidates or [])
        height, width = frame_shape[:2]
        diagonal = max(math.hypot(float(width), float(height)), 1.0)

        next_hypotheses: list[Hypothesis] = []
        for hypothesis in self.hypotheses:
            carried = self._carry(hypothesis, frame_id)
            if carried is not None:
                next_hypotheses.append(carried)
            for candidate in candidates:
                extended = self._extend(
                    hypothesis,
                    candidate,
                    frame_id=frame_id,
                    diagonal=diagonal,
                )
                if extended is not None:
                    next_hypotheses.append(extended)

        # Crucial difference from the single-track gate: every candidate may start a new
        # competing path even while another hypothesis is currently strong.
        next_hypotheses.extend(self._spawn(candidate, frame_id) for candidate in candidates)
        self.hypotheses = self._dedupe(next_hypotheses, diagonal)

        confirmed = [
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.misses == 0
            and hypothesis.last_frame_id == int(frame_id)
            and hypothesis.streak >= self.confirm_frames
            and hypothesis.score >= self.min_emit_score
        ]
        confirmed.sort(key=self._rank_key, reverse=True)

        if not confirmed:
            self.abstentions += 1
            return BallAssociationDecision(
                ball=None,
                source="multi_hypothesis_unconfirmed",
                selected_index=None,
                score=None,
                distance_norm=None,
                gate_norm=None,
                rejected_candidates=len(candidates),
                hold_age_frames=0,
                track_state="DEGRADED" if self.hypotheses else "LOST",
                reacquired=False,
            )

        best = confirmed[0]
        runner_up = confirmed[1] if len(confirmed) > 1 else None
        if runner_up is not None and best.score - runner_up.score < self.emit_margin:
            self.abstentions += 1
            self.ambiguous_abstentions += 1
            return BallAssociationDecision(
                ball=None,
                source="multi_hypothesis_ambiguous",
                selected_index=None,
                score=float(best.score),
                distance_norm=None,
                gate_norm=None,
                rejected_candidates=max(len(candidates) - len(confirmed), 0),
                hold_age_frames=0,
                track_state="DEGRADED",
                reacquired=False,
            )

        selected_index = None
        best_center = np.asarray(best.candidate["center"], dtype=np.float64)
        distances = [
            float(np.linalg.norm(np.asarray(candidate["center"], dtype=np.float64) - best_center))
            for candidate in candidates
        ]
        if distances:
            selected_index = int(min(range(len(distances)), key=lambda idx: distances[idx]))

        ball = dict(best.candidate)
        ball["confidence"] = float(best.score)
        ball["association_score"] = float(best.score)
        ball["interpolated"] = False
        ball["source"] = "multi_hypothesis_confirmed"
        return BallAssociationDecision(
            ball=ball,
            source="multi_hypothesis_confirmed",
            selected_index=selected_index,
            score=float(best.score),
            distance_norm=None,
            gate_norm=None,
            rejected_candidates=max(len(candidates) - 1, 0),
            hold_age_frames=0,
            track_state="TRACKED",
            reacquired=False,
        )


def replay_run(
    run_dir: Path,
    *,
    frame_shape: tuple[int, int, int],
    multi: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = load_jsonl(run_dir / "frames.jsonl")
    tracker: Any = MultiHypothesisTracker() if multi else TemporalBallTracker(reacquire_enabled=False)
    output: list[dict[str, Any]] = []

    for row in rows:
        if not bool(row.get("valid_gameplay")):
            output.append(dict(row))
            continue
        candidates = candidates_from_row(row)
        decision = tracker.select(candidates, int(row.get("frame_id", 0)), frame_shape)
        output.append(decision_row(row, decision))

    diagnostics = {
        "abstentions": int(getattr(tracker, "abstentions", 0)),
        "ambiguous_abstentions": int(getattr(tracker, "ambiguous_abstentions", 0)),
    }
    return output, diagnostics


def print_metrics(label: str, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    p2 = aggregate["p0_2_association"]
    p3 = aggregate["p0_3_trajectory"]
    p4 = aggregate["p0_4_state_robustness"]
    p5 = aggregate["p0_5_system"]
    print(f"\n{label}")
    print("-" * 72)
    print(f"direct coverage:        {p2['direct_correct_coverage_micro_pct']:7.2f}%")
    print(f"direct precision:       {p2['direct_output_precision_micro_pct']:7.2f}%")
    print(f"association miss:       {p2['association_opportunity_miss_rate_micro_pct']:7.2f}%")
    print(f"usable coverage:        {p3['usable_correct_visible_coverage_micro_pct']:7.2f}%")
    print(f"predicted precision:    {p3['predicted_output_precision_micro_pct']:7.2f}%")
    print(f"not-visible false:      {p4['not_visible_false_state_rate_micro_pct']:7.2f}%")
    print(f"catastrophic events:    {p4['catastrophic_error_events']:7d}")
    print(f"unsupported gap P90:    {str(p5['unsupported_gap_p90_frames']):>7} frames")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay frozen Smart candidates through baseline single-track association and "
            "one conservative multi-hypothesis/abstention policy. No inference or GT is "
            "used by the association policy itself."
        )
    )
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--run-root", default="evaluation/runs")
    parser.add_argument("--gt-root", default="evaluation/ball_ground_truth")
    parser.add_argument("--output", default="evaluation/p0_multi_hypothesis_replay")
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
    multi_results: list[dict[str, Any]] = []
    total_abstentions = 0
    total_ambiguous = 0

    for run_id, gt_id in args.pair:
        run_dir = run_root / run_id
        frame_shape = resolve_frame_shape(run_dir, fallback)
        baseline_rows, _ = replay_run(run_dir, frame_shape=frame_shape, multi=False)
        multi_rows, diagnostics = replay_run(run_dir, frame_shape=frame_shape, multi=True)
        total_abstentions += diagnostics["abstentions"]
        total_ambiguous += diagnostics["ambiguous_abstentions"]

        baseline_path = output_root / f"{run_id}_baseline.jsonl"
        multi_path = output_root / f"{run_id}_multi_hypothesis.jsonl"
        write_jsonl(baseline_path, baseline_rows)
        write_jsonl(multi_path, multi_rows)

        baseline = evaluate_pair(
            baseline_path,
            gt_root / gt_id / "annotations.jsonl",
            radius_px=40.0,
            catastrophic_radius_px=160.0,
            timestamp_tolerance_sec=0.001,
        )
        multi_result = evaluate_pair(
            multi_path,
            gt_root / gt_id / "annotations.jsonl",
            radius_px=40.0,
            catastrophic_radius_px=160.0,
            timestamp_tolerance_sec=0.001,
        )
        baseline["run_id"] = run_id
        baseline["gt_run_id"] = gt_id
        multi_result["run_id"] = run_id
        multi_result["gt_run_id"] = gt_id
        baseline_results.append(baseline)
        multi_results.append(multi_result)

    thresholds = GateThresholds()
    baseline_summary = build_summary(
        baseline_results,
        thresholds=thresholds,
        catastrophic_radius_px=160.0,
    )
    multi_summary = build_summary(
        multi_results,
        thresholds=thresholds,
        catastrophic_radius_px=160.0,
    )

    print_metrics("BASELINE REPLAY", baseline_summary)
    print_metrics("CONFIRMED MULTI-HYPOTHESIS", multi_summary)
    print("\nMULTI-HYPOTHESIS DIAGNOSTICS")
    print("-" * 72)
    print(f"abstention frames:       {total_abstentions}")
    print(f"ambiguous abstentions:   {total_ambiguous}")
    print("predicted holds:          disabled by design in this experiment")
    print("hard reacquisition:       disabled")

    (output_root / "baseline_summary.json").write_text(
        __import__("json").dumps(baseline_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "multi_hypothesis_summary.json").write_text(
        __import__("json").dumps(multi_summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
