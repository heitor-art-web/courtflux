from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_gate_d_detector_v5 import PerformanceDetector


def _frame(*, gameplay_present: bool, analysis_eligible: bool):
    return SimpleNamespace(
        court_state=SimpleNamespace(
            gameplay_present=gameplay_present,
            analysis_eligible=analysis_eligible,
        )
    )


def test_limited_gameplay_widens_gate_d_continuity_scope() -> None:
    frame = _frame(gameplay_present=True, analysis_eligible=False)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is True


def test_analysis_eligible_frame_needs_no_widening() -> None:
    frame = _frame(gameplay_present=True, analysis_eligible=True)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is False


def test_non_gameplay_frame_is_not_widened() -> None:
    frame = _frame(gameplay_present=False, analysis_eligible=False)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is False


def test_missing_court_state_is_not_widened() -> None:
    frame = SimpleNamespace(court_state=None)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is False


def main() -> None:
    test_limited_gameplay_widens_gate_d_continuity_scope()
    test_analysis_eligible_frame_needs_no_widening()
    test_non_gameplay_frame_is_not_widened()
    test_missing_court_state_is_not_widened()
    print("game_player_gate_d_detector_v5_scope_test: PASS")


if __name__ == "__main__":
    main()
