from __future__ import annotations

import torch

from services.lab_contract import resolve_jersey_backend
from services.vision.p1_trocr_jersey_analyzer import TrOCRSmallPrintedJerseyAnalyzer


def test_strict_ocr_parser() -> None:
    assert TrOCRSmallPrintedJerseyAnalyzer.parse_ocr_number("23") == ("23", "parsed")
    assert TrOCRSmallPrintedJerseyAnalyzer.parse_ocr_number(" 07 ") == ("7", "parsed")
    assert TrOCRSmallPrintedJerseyAnalyzer.parse_ocr_number("00") == ("00", "parsed")
    assert TrOCRSmallPrintedJerseyAnalyzer.parse_ocr_number("jersey 23") == (
        None,
        "no_valid_number",
    )
    assert TrOCRSmallPrintedJerseyAnalyzer.parse_ocr_number("123") == (
        None,
        "no_valid_number",
    )
    assert TrOCRSmallPrintedJerseyAnalyzer.parse_ocr_number("") == (
        None,
        "no_text_output",
    )


def test_sequence_confidence_ignores_special_tokens() -> None:
    sequence = torch.tensor([0, 1, 2])
    scores = [
        torch.tensor([[0.0, 10.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 10.0]]),
    ]
    confidence = TrOCRSmallPrintedJerseyAnalyzer._sequence_confidence(
        sequence,
        scores,
        special_token_ids={2},
    )
    assert confidence > 0.99


def test_direct_sentencepiece_decoder_uses_xlm_roberta_offset() -> None:
    class FakeSentencePiece:
        pieces = ["<unk>", "<s>", "</s>", "▁2", "3"]

        def get_piece_size(self) -> int:
            return len(self.pieces)

        def IdToPiece(self, index: int) -> str:
            return self.pieces[index]

    analyzer = object.__new__(TrOCRSmallPrintedJerseyAnalyzer)
    analyzer.sp_model = FakeSentencePiece()
    analyzer.fairseq_offset = 1
    analyzer.mask_token_id = analyzer.sp_model.get_piece_size() + analyzer.fairseq_offset

    # decoder-start/eos are fairseq id 2; real SPM ids are shifted by +1.
    assert analyzer._decode_sequence(torch.tensor([2, 4, 5, 2])) == "23"
    assert analyzer._decode_sequence(torch.tensor([2, 3, 2])) == "UNKNOWN"
    assert analyzer._decode_sequence(torch.tensor([2, analyzer.mask_token_id, 2])) == "UNKNOWN"


def test_backend_aliases() -> None:
    assert resolve_jersey_backend("trocr") == "trocr_small_printed"
    assert resolve_jersey_backend("trocr-small-printed") == "trocr_small_printed"
    assert resolve_jersey_backend("trocr_small_printed") == "trocr_small_printed"


def main() -> int:
    test_strict_ocr_parser()
    test_sequence_confidence_ignores_special_tokens()
    test_direct_sentencepiece_decoder_uses_xlm_roberta_offset()
    test_backend_aliases()
    print("p1_trocr_jersey_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
