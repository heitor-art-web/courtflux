from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.game_player_gate_d_detector_v5 import PerformanceDetector


def _frame(
    *,
    gameplay_present: bool,
    analysis_eligible: bool,
    players: list[dict] | None = None,
    frame_id: int = 12,
):
    return SimpleNamespace(
        frame_id=frame_id,
        court_state=SimpleNamespace(
            gameplay_present=gameplay_present,
            analysis_eligible=analysis_eligible,
        ),
        players=list(players or []),
        diagnostics={"detector": {}},
    )


def test_limited_gameplay_widens_gate_d_continuity_scope() -> None:
    frame = _frame(gameplay_present=True, analysis_eligible=False)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is True
    assert PerformanceDetector._gate_d_continuity_mode(frame) == "gameplay_limited"


def test_analysis_eligible_frame_needs_no_widening() -> None:
    frame = _frame(gameplay_present=True, analysis_eligible=True)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is False
    assert PerformanceDetector._gate_d_continuity_mode(frame) == "gameplay_analysis"


def test_non_gameplay_frame_freezes_continuity_scope() -> None:
    frame = _frame(gameplay_present=False, analysis_eligible=False)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is False
    assert PerformanceDetector._gate_d_continuity_mode(frame) == "frozen_non_gameplay"


def test_missing_court_state_uses_legacy_scope() -> None:
    frame = SimpleNamespace(court_state=None)
    assert PerformanceDetector._gate_d_widen_continuity_scope(frame) is False
    assert PerformanceDetector._gate_d_continuity_mode(frame) == "legacy"


def test_limited_gameplay_keeps_only_explicit_positive_on_court_players() -> None:
    frame = _frame(
        gameplay_present=True,
        analysis_eligible=False,
        players=[
            {"track_id": 1, "active_on_court": True},
            {"track_id": 2, "active_on_court": None},
            {"track_id": 3, "active_on_court": False},
            {"track_id": 4},
        ],
    )
    eligible = PerformanceDetector._gate_d_limited_players(frame)
    assert [row["track_id"] for row in eligible] == [1]


class _FreezeOnlyResolver:
    def __init__(self) -> None:
        self.frozen_frames: list[int] = []

    def freeze_non_gameplay(self, frame_id: int) -> None:
        self.frozen_frames.append(int(frame_id))

    def stoppage_diagnostics(self) -> dict[str, object]:
        return {
            "stoppage_gap_suspension": True,
            "completed_frozen_ranges": 0,
            "active_frozen_range": (
                None if not self.frozen_frames else [self.frozen_frames[0], self.frozen_frames[-1]]
            ),
            "completed_frozen_frames": 0,
        }

    def prepare_team_stability(self, _confirmations):
        raise AssertionError("NON_GAMEPLAY must not prepare identity association state")

    def clear_team_stability(self):
        raise AssertionError("NON_GAMEPLAY must not clear identity association state")

    def observe(self, **_kwargs):
        raise AssertionError("NON_GAMEPLAY must not create/update an identity observation")

    def prepare_frame(self, **_kwargs):
        raise AssertionError("NON_GAMEPLAY must not run appearance preflight")


def test_non_gameplay_observation_path_only_advances_stoppage_clock() -> None:
    detector = object.__new__(PerformanceDetector)
    resolver = _FreezeOnlyResolver()
    detector.game_player_resolver = resolver
    frame = _frame(
        gameplay_present=False,
        analysis_eligible=False,
        players=[{"track_id": 7, "active_on_court": True}],
        frame_id=77,
    )
    frame.diagnostics["detector"]["team_predictions"] = [
        {"track_id": 7, "recent_reversal_confirmed": True}
    ]

    detector._observe_game_players(frame)

    assert resolver.frozen_frames == [77]
    assert frame.diagnostics["detector"]["game_player_predictions"] == []
    runtime = frame.diagnostics["detector"]["game_player_runtime"]
    assert runtime["continuity_mode"] == "frozen_non_gameplay"
    assert runtime["continuity_frozen"] is True
    assert runtime["non_gameplay_freezes_resolver"] is True
    assert runtime["stoppage_gap_suspension"] is True
    assert runtime["active_frozen_range"] == [77, 77]


def main() -> None:
    test_limited_gameplay_widens_gate_d_continuity_scope()
    test_analysis_eligible_frame_needs_no_widening()
    test_non_gameplay_frame_freezes_continuity_scope()
    test_missing_court_state_uses_legacy_scope()
    test_limited_gameplay_keeps_only_explicit_positive_on_court_players()
    test_non_gameplay_observation_path_only_advances_stoppage_clock()
    print("game_player_gate_d_detector_v5_scope_test: PASS")


if __name__ == "__main__":
    main()
