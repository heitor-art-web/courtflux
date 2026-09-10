from __future__ import annotations

import json
import tempfile

import numpy as np

from services.vision.roboflow_jersey_analyzer import RoboflowJerseyAnalyzer


def _expect(response, expected_number, expected_reason):
    number, reason = RoboflowJerseyAnalyzer.parse_number(response)
    assert number == expected_number, (response, number, expected_number)
    assert reason == expected_reason, (response, reason, expected_reason)


class _FakeLMMClient:
    def __init__(self) -> None:
        self.calls = []

    def infer_lmm(self, **kwargs):
        self.calls.append(kwargs)
        return {"response": {"text": "23"}}


def _test_lmm_call_contract() -> None:
    # Bypass __init__ so the test is deterministic and never needs a live Roboflow server.
    analyzer = RoboflowJerseyAnalyzer.__new__(RoboflowJerseyAnalyzer)
    analyzer.model_id = "basketball-jersey-numbers-ocr/7"
    analyzer.api_url = "http://localhost:9001"
    analyzer.prompt = "read jersey"
    analyzer.client = _FakeLMMClient()
    analyzer.track_crop_buffers = {
        17: [np.zeros((72, 40, 3), dtype=np.uint8)],
    }
    analyzer.rosters = {
        "Test Team": {
            "roster": {
                "23": {"name": "Test Player"},
            }
        }
    }

    result = analyzer.predict_jersey(17, "Test Team")
    assert result["accepted"] is True
    assert result["candidate_number"] == "23"
    assert result["latency_ms"] >= 0.0
    assert len(analyzer.client.calls) == 1
    call = analyzer.client.calls[0]
    assert call["model_id"] == "basketball-jersey-numbers-ocr/7"
    assert call["model_id_in_path"] is True
    assert call["max_new_tokens"] == 16


def _test_serverless_rejected() -> None:
    config = {"metadata": {}, "teams": {}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
        json.dump(config, handle)
        handle.flush()
        try:
            RoboflowJerseyAnalyzer(
                profiling_path=handle.name,
                api_url="https://serverless.roboflow.com",
                api_key="placeholder",
            )
        except RuntimeError as exc:
            assert "Serverless/Hosted" in str(exc)
        else:
            raise AssertionError("serverless jersey endpoint should be rejected")


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

    # Metadata strings must never be interpreted as jersey OCR output.
    _expect(
        {"model_id": "basketball-jersey-numbers-ocr/7", "latency_ms": "23"},
        None,
        "no_text_output",
    )
    _expect(
        {
            "model_id": "basketball-jersey-numbers-ocr/7",
            "payload": {"response": {"text": "32"}},
        },
        "32",
        "parsed",
    )

    # Repeated equivalent evidence is unambiguous.
    _expect({"result": ["23", "jersey 23"]}, "23", "parsed")
    _test_lmm_call_contract()
    _test_serverless_rejected()

    print("jersey_recognition_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
