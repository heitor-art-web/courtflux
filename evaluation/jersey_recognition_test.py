from __future__ import annotations

from services.vision.roboflow_jersey_analyzer import RoboflowJerseyAnalyzer


def _expect(response, expected_number, expected_reason):
    number, reason = RoboflowJerseyAnalyzer.parse_number(response)
    assert number == expected_number, (response, number, expected_number)
    assert reason == expected_reason, (response, reason, expected_reason)


def main() -> int:
    _expect({"text": "23"}, "23", "parsed")
    _expect({"output": {"text": "Jersey number: 7"}}, "7", "parsed")
    _expect([{"generated_text": "00"}], "00", "parsed")
    _expect({"prediction": "07"}, "7", "parsed")
    _expect({"prediction": "011"}, None, "out_of_range_or_malformed")
    _expect({"prediction": "3000"}, None, "out_of_range_or_malformed")
    _expect({"prediction": "23 or 24"}, None, "ambiguous_numeric_output")
    _expect({"prediction": "number not visible"}, None, "no_valid_number")
    _expect({}, None, "no_text_output")

    # Repeated equivalent evidence is unambiguous.
    _expect({"result": ["23", "jersey 23"]}, "23", "parsed")

    print("jersey_recognition_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
