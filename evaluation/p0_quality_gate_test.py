from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.p0_quality_gate import (
    GateThresholds,
    build_summary,
    evaluate_pair,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def candidate(center: tuple[float, float], confidence: float = 0.9) -> dict:
    x, y = center
    return {
        "center": [x, y],
        "box": [x - 5, y - 5, x + 5, y + 5],
        "confidence": confidence,
    }


def production_row(
    frame_id: int,
    *,
    output_kind: str,
    output_center: tuple[float, float] | None,
    candidates: list[dict],
) -> dict:
    timestamp = frame_id / 30.0
    if output_kind == "direct":
        source = "detector_associated"
        selected_ball = True
        selected_ball_center = list(output_center) if output_center is not None else None
        selected_index = 0
        trajectory_center = list(output_center) if output_center is not None else None
        hold_age = 0
    elif output_kind == "predicted":
        source = "predicted_hold"
        selected_ball = False
        selected_ball_center = None
        selected_index = None
        trajectory_center = list(output_center) if output_center is not None else None
        hold_age = 1
    else:
        source = "missing"
        selected_ball = False
        selected_ball_center = None
        selected_index = None
        trajectory_center = None
        hold_age = 0

    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp,
        "valid_gameplay": True,
        "selected_ball": selected_ball,
        "selected_ball_center": selected_ball_center,
        "ball_source": source,
        "ball_association": {
            "source": source,
            "selected_index": selected_index,
            "trajectory_center": trajectory_center,
            "hold_age_frames": hold_age,
            "reacquired": False,
            "candidates": candidates,
        },
    }


def truth_row(
    frame_id: int,
    *,
    visible: bool,
    point: tuple[float, float] = (100.0, 100.0),
) -> dict:
    row = {
        "timestamp_sec": frame_id / 30.0,
        "visible": visible,
    }
    if visible:
        row["point"] = [float(point[0]), float(point[1])]
    return row


def evaluate_rows(production: list[dict], truth: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = root / "frames.jsonl"
        annotations = root / "annotations.jsonl"
        write_jsonl(frames, production)
        write_jsonl(annotations, truth)
        return evaluate_pair(
            frames,
            annotations,
            radius_px=40.0,
            catastrophic_radius_px=160.0,
            timestamp_tolerance_sec=0.001,
        )


def test_all_p0_stages_perfect_system() -> None:
    production: list[dict] = []
    truth: list[dict] = []
    for frame_id in range(10):
        gt = (100.0 + frame_id * 3.0, 100.0)
        truth.append(truth_row(frame_id, visible=True, point=gt))
        kind = "direct" if frame_id < 8 else "predicted"
        production.append(
            production_row(
                frame_id,
                output_kind=kind,
                output_center=gt,
                candidates=[candidate(gt)],
            )
        )

    for frame_id in range(10, 12):
        truth.append(truth_row(frame_id, visible=False))
        production.append(
            production_row(
                frame_id,
                output_kind="missing",
                output_center=None,
                candidates=[],
            )
        )

    result = evaluate_rows(production, truth)
    metrics = result["metrics"]
    assert metrics["p0_1_detector_candidates"]["visible_candidate_recall_pct"] == 100.0
    assert metrics["p0_2_association"]["direct_correct_coverage_pct"] == 80.0
    assert metrics["p0_2_association"]["association_opportunity_miss_rate_pct"] == 0.0
    assert metrics["p0_3_trajectory"]["usable_correct_visible_coverage_pct"] == 100.0
    assert metrics["p0_3_trajectory"]["coverage_lift_over_direct_pp"] == 20.0
    assert metrics["p0_4_state_robustness"]["not_visible_false_state_rate_pct"] == 0.0
    assert metrics["p0_4_state_robustness"]["catastrophic_error_events"] == 0
    assert metrics["p0_5_system"]["concrete_output_precision_pct"] == 100.0
    assert metrics["p0_5_system"]["unsupported_gap_p90_frames"] is None

    summary = build_summary(
        [result],
        thresholds=GateThresholds(),
        catastrophic_radius_px=160.0,
    )
    assert summary["gate"]["status"] == "PASS"


def test_association_miss_gap_and_catastrophic_event() -> None:
    production: list[dict] = []
    truth: list[dict] = []

    # Correct anchor.
    truth.append(truth_row(0, visible=True, point=(100.0, 100.0)))
    production.append(
        production_row(
            0,
            output_kind="direct",
            output_center=(100.0, 100.0),
            candidates=[candidate((100.0, 100.0))],
        )
    )

    # True candidate exists, but association emits a catastrophic alternative.
    truth.append(truth_row(1, visible=True, point=(105.0, 100.0)))
    wrong = candidate((400.0, 400.0), confidence=0.95)
    correct = candidate((105.0, 100.0), confidence=0.40)
    production.append(
        production_row(
            1,
            output_kind="direct",
            output_center=(400.0, 400.0),
            candidates=[wrong, correct],
        )
    )

    # Two additional unsupported visible frames.
    for frame_id in (2, 3):
        gt = (105.0 + frame_id * 3.0, 100.0)
        truth.append(truth_row(frame_id, visible=True, point=gt))
        production.append(
            production_row(
                frame_id,
                output_kind="missing",
                output_center=None,
                candidates=[candidate(gt)],
            )
        )

    # Recovery closes a 3-frame unsupported gap.
    truth.append(truth_row(4, visible=True, point=(117.0, 100.0)))
    production.append(
        production_row(
            4,
            output_kind="direct",
            output_center=(117.0, 100.0),
            candidates=[candidate((117.0, 100.0))],
        )
    )

    # Not-visible false output must count against state robustness and precision.
    truth.append(truth_row(5, visible=False))
    production.append(
        production_row(
            5,
            output_kind="predicted",
            output_center=(120.0, 100.0),
            candidates=[],
        )
    )

    result = evaluate_rows(production, truth)
    metrics = result["metrics"]
    assert metrics["p0_1_detector_candidates"]["visible_candidate_recall_pct"] == 100.0
    assert metrics["p0_2_association"]["association_opportunity_miss_rate_pct"] == 60.0
    assert metrics["p0_4_state_robustness"]["catastrophic_error_events"] == 1
    assert metrics["p0_4_state_robustness"]["not_visible_false_state_rate_pct"] == 100.0
    assert metrics["p0_5_system"]["unsupported_gap_max_frames"] == 3

    summary = build_summary(
        [result],
        thresholds=GateThresholds(),
        catastrophic_radius_px=160.0,
    )
    assert summary["gate"]["status"] == "FAIL"
    assert not summary["gate"]["blocking_checks"]["catastrophic_error_events"]["pass"]
    assert not summary["gate"]["blocking_checks"]["not_visible_false_state"]["pass"]


def test_direct_detector_target_is_supporting_not_blocking() -> None:
    # Build a synthetic aggregate that passes every blocking condition but has weak
    # direct candidate recall. The detector target must remain a supporting check.
    result = {
        "counts": {
            key: 0
            for key in (
                "labelled_frames",
                "visible_frames",
                "not_visible_frames",
                "candidate_frames",
                "candidate_instances",
                "candidate_true_instances",
                "candidate_false_instances",
                "visible_candidate_correct_frames",
                "not_visible_candidate_frames",
                "direct_output_frames",
                "predicted_output_frames",
                "missing_output_frames",
                "visible_direct_correct_frames",
                "visible_direct_wrong_frames",
                "visible_predicted_correct_frames",
                "visible_predicted_wrong_frames",
                "visible_missing_frames",
                "visible_usable_correct_frames",
                "concrete_output_frames",
                "concrete_correct_frames",
                "not_visible_false_state_frames",
                "not_visible_direct_false_frames",
                "not_visible_predicted_false_frames",
                "association_opportunity_frames",
                "association_opportunity_missed_frames",
                "catastrophic_error_frames",
                "catastrophic_error_events",
                "reacquisition_events",
            )
        },
        "unsupported_gaps_frames": [],
        "metrics": {
            "p0_5_system": {"visible_usable_correct_coverage_pct": 100.0}
        },
    }
    counts = result["counts"]
    counts.update(
        {
            "labelled_frames": 110,
            "visible_frames": 100,
            "not_visible_frames": 10,
            "visible_candidate_correct_frames": 50,
            "candidate_frames": 50,
            "candidate_instances": 50,
            "candidate_true_instances": 50,
            "visible_direct_correct_frames": 50,
            "visible_predicted_correct_frames": 50,
            "visible_usable_correct_frames": 100,
            "direct_output_frames": 50,
            "predicted_output_frames": 50,
            "concrete_output_frames": 100,
            "concrete_correct_frames": 100,
        }
    )

    summary = build_summary(
        [result],
        thresholds=GateThresholds(),
        catastrophic_radius_px=160.0,
    )
    assert summary["gate"]["status"] == "PASS"
    supporting = summary["gate"]["supporting_checks"]["direct_candidate_recall"]
    assert supporting["value"] == 50.0
    assert not supporting["pass"]
    assert supporting["blocking"] is False


def main() -> None:
    test_all_p0_stages_perfect_system()
    test_association_miss_gap_and_catastrophic_event()
    test_direct_detector_target_is_supporting_not_blocking()
    print("[PASS] P0 quality gate metrics")


if __name__ == "__main__":
    main()
