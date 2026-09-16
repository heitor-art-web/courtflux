from __future__ import annotations

from evaluation.jersey_recon_benchmark import Row, roi_iou, score


def test_roi_iou_exact_match() -> None:
    assert roi_iou([0.1, 0.2, 0.5, 0.7], [0.1, 0.2, 0.5, 0.7]) == 1.0


def test_score_separates_visibility_recognition_and_track_identity() -> None:
    rows = [
        Row(
            track_id="a",
            gt_number="21",
            visible=True,
            oracle_roi=[0.30, 0.24, 0.68, 0.58],
            auto_roi=[0.32, 0.25, 0.69, 0.59],
            oracle_prediction="21",
            auto_prediction="21",
            final_track_prediction="21",
        ),
        Row(
            track_id="b",
            gt_number="4",
            visible=True,
            oracle_roi=[0.34, 0.20, 0.62, 0.52],
            auto_roi=[0.10, 0.10, 0.30, 0.30],
            oracle_prediction="4",
            auto_prediction="8",
            final_track_prediction="4",
        ),
        Row(
            track_id="c",
            gt_number="13",
            visible=False,
            oracle_roi=None,
            auto_roi=None,
            oracle_prediction=None,
            auto_prediction=None,
            final_track_prediction=None,
        ),
    ]

    result = score(rows, iou_threshold=0.50)

    assert result["tracks"]["visibility_rate"] == 0.666667
    assert result["oracle_recognizer"]["accuracy_on_visible"] == 1.0
    assert result["automatic_recognizer"]["accuracy_on_visible"] == 0.5
    assert result["oracle_minus_auto_accuracy_gap"] == 0.5
    assert result["final_track_identification"]["accuracy"] == 0.666667


def main() -> None:
    test_roi_iou_exact_match()
    test_score_separates_visibility_recognition_and_track_identity()
    print("jersey_recon_benchmark_test: PASS")


if __name__ == "__main__":
    main()
