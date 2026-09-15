from __future__ import annotations

import numpy as np

from services.lab_contract import resolve_jersey_backend
from services.vision.p1_parseq_jersey_analyzer import SoccerNetPARSeqJerseyAnalyzer


def test_strict_ocr_parser() -> None:
    assert SoccerNetPARSeqJerseyAnalyzer.parse_ocr_number("23") == ("23", "parsed")
    assert SoccerNetPARSeqJerseyAnalyzer.parse_ocr_number(" 07 ") == ("7", "parsed")
    assert SoccerNetPARSeqJerseyAnalyzer.parse_ocr_number("00") == ("00", "parsed")
    assert SoccerNetPARSeqJerseyAnalyzer.parse_ocr_number("jersey 23") == (
        None,
        "no_valid_number",
    )
    assert SoccerNetPARSeqJerseyAnalyzer.parse_ocr_number("123") == (
        None,
        "no_valid_number",
    )


def test_preprocessing_contract() -> None:
    crop = np.full((80, 40, 3), 127, dtype=np.uint8)
    tensor = SoccerNetPARSeqJerseyAnalyzer._prepare_crop(crop)
    assert tuple(tensor.shape) == (3, 32, 128)
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_backend_aliases() -> None:
    assert resolve_jersey_backend("parseq") == "parseq_soccernet"
    assert resolve_jersey_backend("soccernet_parseq") == "parseq_soccernet"
    assert resolve_jersey_backend("parseq-soccernet") == "parseq_soccernet"
    assert resolve_jersey_backend("parseq_soccernet") == "parseq_soccernet"


def main() -> int:
    test_strict_ocr_parser()
    test_preprocessing_contract()
    test_backend_aliases()
    print("p1_parseq_jersey_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
