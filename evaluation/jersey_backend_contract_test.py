from __future__ import annotations

from pathlib import Path

from services.lab_contract import (
    LAB_DEFAULTS,
    resolve_jersey_backend,
    validate_jersey_backend_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_parseq_is_canonical_default() -> None:
    assert LAB_DEFAULTS["jersey_backend"] == "parseq_soccernet"
    assert resolve_jersey_backend("parseq") == "parseq_soccernet"
    assert resolve_jersey_backend("soccernet_parseq") == "parseq_soccernet"


def test_backend_contract_fails_on_silent_fallback() -> None:
    assert (
        validate_jersey_backend_contract("parseq_soccernet", "parseq_soccernet")
        == "parseq_soccernet"
    )
    try:
        validate_jersey_backend_contract("parseq_soccernet", "best_temporal")
    except RuntimeError as exc:
        assert "requested='parseq_soccernet'" in str(exc)
        assert "runtime built='best_temporal'" in str(exc)
    else:
        raise AssertionError("silent Jersey backend fallback must fail closed")


def test_worker_propagates_backend_to_analyzer_process() -> None:
    source = (PROJECT_ROOT / "services" / "worker_api.py").read_text(encoding="utf-8")
    assert "jersey_backend: str" in source
    assert "resolve_jersey_backend" in source
    assert 'child_env["BASKETBALL_JERSEY_BACKEND"] = selected_backend' in source
    assert "[WORKER] jersey_backend=" in source


def test_factory_uses_shared_contract_and_verifies_actual_backend() -> None:
    source = (PROJECT_ROOT / "services" / "vision" / "jersey_backend_factory.py").read_text(
        encoding="utf-8"
    )
    assert "requested = resolve_jersey_backend()" in source
    assert "validate_jersey_backend_contract" in source
    assert "contract=verified" in source


def main() -> int:
    test_parseq_is_canonical_default()
    test_backend_contract_fails_on_silent_fallback()
    test_worker_propagates_backend_to_analyzer_process()
    test_factory_uses_shared_contract_and_verifies_actual_backend()
    print("jersey_backend_contract_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
