from __future__ import annotations

import numpy as np

from services.vision.p1_parseq_jersey_analyzer import SoccerNetPARSeqJerseyAnalyzer


def _row(
    number: str | None,
    confidence: float,
    legibility: float,
    frame_id: int,
    *,
    roi_prior: float = 1.0,
) -> dict:
    return {
        "number": number,
        "confidence": confidence,
        "legibility_score": legibility,
        "frame_id": frame_id,
        "roi_prior": roi_prior,
    }


def test_consensus_accepts_repeated_number_across_frames() -> None:
    result = SoccerNetPARSeqJerseyAnalyzer._multi_frame_consensus(
        [
            _row("21", 0.90, 0.92, 10),
            _row("21", 0.82, 0.86, 11),
            _row("14", 0.70, 0.60, 12),
            _row(None, 0.30, 0.80, 13),
        ],
        min_support=2,
        min_share=0.60,
    )
    assert result["number"] == "21"
    assert result["reason"] == "parsed"
    assert result["support"] == 2
    assert result["share"] >= 0.60


def test_consensus_abstains_on_conflict() -> None:
    result = SoccerNetPARSeqJerseyAnalyzer._multi_frame_consensus(
        [
            _row("10", 0.95, 0.90, 10),
            _row("27", 0.93, 0.91, 11),
            _row("14", 0.92, 0.89, 12),
            _row("8", 0.90, 0.88, 13),
        ],
        min_support=2,
        min_share=0.60,
    )
    assert result["number"] is None
    assert result["reason"] == "low_multiframe_consensus"
    assert result["support"] == 1


def test_multiple_rois_on_one_frame_do_not_fake_temporal_support() -> None:
    result = SoccerNetPARSeqJerseyAnalyzer._multi_frame_consensus(
        [
            _row("10", 0.99, 0.95, 10),
            _row("10", 0.98, 0.94, 10),
            _row("10", 0.97, 0.93, 10),
        ],
        min_support=2,
        min_share=0.60,
    )
    assert result["number"] is None
    assert result["reason"] == "low_multiframe_consensus"
    assert result["support"] == 1


def test_consensus_requires_support_even_with_one_confident_crop() -> None:
    result = SoccerNetPARSeqJerseyAnalyzer._multi_frame_consensus(
        [
            _row("10", 0.99, 0.99, 10),
            _row(None, 0.0, 0.70, 11),
            _row(None, 0.0, 0.68, 12),
        ],
        min_support=2,
        min_share=0.60,
    )
    assert result["number"] is None
    assert result["reason"] == "low_multiframe_consensus"
    assert result["support"] == 1


def test_localized_roi_crop_is_smaller_than_torso() -> None:
    crop = np.zeros((100, 80, 3), dtype=np.uint8)
    roi = SoccerNetPARSeqJerseyAnalyzer._roi_crop(crop, 0.16, 0.04, 0.84, 0.48)
    assert roi is not None
    assert roi.shape[0] < crop.shape[0]
    assert roi.shape[1] < crop.shape[1]
    assert roi.shape[0] >= 12
    assert roi.shape[1] >= 12


def main() -> None:
    test_consensus_accepts_repeated_number_across_frames()
    test_consensus_abstains_on_conflict()
    test_multiple_rois_on_one_frame_do_not_fake_temporal_support()
    test_consensus_requires_support_even_with_one_confident_crop()
    test_localized_roi_crop_is_smaller_than_torso()
    print("jersey_recon_multiframe_test: PASS")


if __name__ == "__main__":
    main()
