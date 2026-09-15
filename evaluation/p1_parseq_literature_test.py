from __future__ import annotations

import numpy as np

from services.lab_contract import resolve_jersey_backend
from services.vision.p1_parseq_literature_analyzer import LiteraturePARSeqJerseyAnalyzer


def test_backend_aliases() -> None:
    assert resolve_jersey_backend("parseq_lit") == "parseq_soccernet_lit"
    assert resolve_jersey_backend("parseq-lit") == "parseq_soccernet_lit"
    assert resolve_jersey_backend("parseq_literature") == "parseq_soccernet_lit"
    assert resolve_jersey_backend("parseq_soccernet_lit") == "parseq_soccernet_lit"


def test_legibility_preprocessing_contract() -> None:
    crop = np.full((120, 64, 3), 127, dtype=np.uint8)
    tensor = LiteraturePARSeqJerseyAnalyzer._prepare_legibility(crop)
    assert tuple(tensor.shape) == (3, 256, 256)
    assert np.isfinite(tensor.numpy()).all()


def test_roi_proposals_are_localized() -> None:
    player = np.full((200, 100, 3), 127, dtype=np.uint8)
    proposals = LiteraturePARSeqJerseyAnalyzer._roi_proposals(player)
    assert [name for name, _ in proposals] == [
        "torso_wide",
        "number_center",
        "number_upper",
    ]
    assert all(roi.size > 0 for _, roi in proposals)
    assert all(roi.shape[0] < player.shape[0] for _, roi in proposals)
    assert all(roi.shape[1] < player.shape[1] for _, roi in proposals)


def test_multi_frame_consensus_accepts_repeated_visual_read() -> None:
    frame_evidence = [
        {
            "frame_id": 10,
            "legibility": 0.91,
            "crop_quality": 0.90,
            "reads": [
                {"number": "23", "confidence": 0.92},
                {"number": "23", "confidence": 0.88},
                {"number": "28", "confidence": 0.71},
            ],
        },
        {
            "frame_id": 11,
            "legibility": 0.88,
            "crop_quality": 0.86,
            "reads": [
                {"number": "23", "confidence": 0.90},
                {"number": "23", "confidence": 0.87},
            ],
        },
        {
            "frame_id": 12,
            "legibility": 0.83,
            "crop_quality": 0.84,
            "reads": [{"number": "14", "confidence": 0.78}],
        },
    ]
    result = LiteraturePARSeqJerseyAnalyzer._aggregate_frame_reads(
        frame_evidence,
        min_confidence=0.50,
        min_frame_support=2,
        min_vote_share=0.60,
        proposal_margin=0.05,
    )
    assert result["number"] == "23"
    assert result["reason"] == "parsed_temporal_roi_consensus"
    assert result["best_support"] == 2
    assert result["best_share"] > 0.60


def test_single_frame_hallucination_is_not_enough() -> None:
    frame_evidence = [
        {
            "frame_id": 20,
            "reads": [{"number": "10", "confidence": 0.99}],
        },
        {
            "frame_id": 21,
            "reads": [{"number": "14", "confidence": 0.96}],
        },
    ]
    result = LiteraturePARSeqJerseyAnalyzer._aggregate_frame_reads(
        frame_evidence,
        min_confidence=0.50,
        min_frame_support=2,
        min_vote_share=0.60,
        proposal_margin=0.05,
    )
    assert result["number"] is None
    assert result["reason"] == "low_visual_consensus_support"


def test_same_frame_roi_duplicates_count_once() -> None:
    frame_evidence = [
        {
            "frame_id": 30,
            "reads": [
                {"number": "10", "confidence": 0.99},
                {"number": "10", "confidence": 0.98},
                {"number": "10", "confidence": 0.97},
            ],
        },
        {
            "frame_id": 31,
            "reads": [{"number": "14", "confidence": 0.96}],
        },
    ]
    result = LiteraturePARSeqJerseyAnalyzer._aggregate_frame_reads(
        frame_evidence,
        min_confidence=0.50,
        min_frame_support=2,
        min_vote_share=0.60,
        proposal_margin=0.05,
    )
    assert result["number"] is None
    assert len(result["frame_winners"]) == 2


def main() -> int:
    test_backend_aliases()
    test_legibility_preprocessing_contract()
    test_roi_proposals_are_localized()
    test_multi_frame_consensus_accepts_repeated_visual_read()
    test_single_frame_hallucination_is_not_enough()
    test_same_frame_roi_duplicates_count_once()
    print("p1_parseq_literature_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
