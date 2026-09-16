from __future__ import annotations

from services.vision.p1_parseq_jersey_analyzer import SoccerNetPARSeqJerseyAnalyzer


def _row(number: str | None, confidence: float, legibility: float) -> dict:
    return {
        "number": number,
        "confidence": confidence,
        "legibility_score": legibility,
    }


def test_consensus_accepts_repeated_number() -> None:
    result = SoccerNetPARSeqJerseyAnalyzer._multi_frame_consensus(
        [
            _row("21", 0.90, 0.92),
            _row("21", 0.82, 0.86),
            _row("14", 0.70, 0.60),
            _row(None, 0.30, 0.80),
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
            _row("10", 0.95, 0.90),
            _row("27", 0.93, 0.91),
            _row("14", 0.92, 0.89),
            _row("8", 0.90, 0.88),
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
            _row("10", 0.99, 0.99),
            _row(None, 0.0, 0.70),
            _row(None, 0.0, 0.68),
        ],
        min_support=2,
        min_share=0.60,
    )
    assert result["number"] is None
    assert result["reason"] == "low_multiframe_consensus"
    assert result["support"] == 1


def main() -> None:
    test_consensus_accepts_repeated_number()
    test_consensus_abstains_on_conflict()
    test_consensus_requires_support_even_with_one_confident_crop()
    print("jersey_recon_multiframe_test: PASS")


if __name__ == "__main__":
    main()
