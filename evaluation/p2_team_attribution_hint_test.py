from types import SimpleNamespace

from core.basketball_event import BallControlState
from services.p2.engine_v1 import P2EngineV1


def _frame(*, active_on_court, confidence=0.94, margin=0.88):
    return SimpleNamespace(
        players=[
            {
                "track_id": 24,
                "active_on_court": active_on_court,
                "team": None,
                "team_confidence": 0.0,
            }
        ],
        diagnostics={
            "detector": {
                "team_predictions": [
                    {
                        "track_id": 24,
                        "rescored": True,
                        "fused_best_team": "team_b",
                        "fused_confidence": confidence,
                        "fused_margin": margin,
                    }
                ]
            }
        },
    )


def test_fresh_hint_requires_explicit_on_court_support() -> None:
    engine = P2EngineV1()

    trusted = engine._team_attribution_hints(_frame(active_on_court=True))
    unknown_surface = engine._team_attribution_hints(_frame(active_on_court=None))

    assert trusted == {24: ("team_b", 0.94)}
    assert unknown_surface == {}


def test_fresh_hint_keeps_existing_team_thresholds() -> None:
    engine = P2EngineV1()

    weak_confidence = engine._team_attribution_hints(
        _frame(active_on_court=True, confidence=0.64, margin=0.88)
    )
    weak_margin = engine._team_attribution_hints(
        _frame(active_on_court=True, confidence=0.94, margin=0.17)
    )

    assert weak_confidence == {}
    assert weak_margin == {}


def test_public_diagnostics_can_fill_unknown_without_mutating_internal_team() -> None:
    engine = P2EngineV1()
    engine.possession_tracker.current_track_id = 24
    engine.possession_tracker.current_team_id = None
    engine.possession_tracker.current_confidence = 0.8
    engine.possession_tracker.control_state = BallControlState.POSSESSED

    frame = _frame(active_on_court=True)
    hints = engine._team_attribution_hints(frame)
    engine._write_diagnostics(
        frame_state=frame,
        emitted=[],
        semantic_evidence=[],
        team_evidence={},
        team_hints=hints,
        engine_eligible=True,
    )

    possession = frame.diagnostics["p2"]["possession"]
    assert possession["current_team_id"] == "team_b"
    assert possession["team_attribution_fallback_applied"] is True
    assert possession["team_attribution_fallback_source"] == "fresh_on_court_team_hint"
    assert engine.possession_tracker.current_team_id is None


def test_shot_actor_lookup_can_use_hint_without_entering_possession_evidence() -> None:
    engine = P2EngineV1()
    frame = _frame(active_on_court=True)
    hints = engine._team_attribution_hints(frame)

    team_id, confidence, source = engine._attributed_team_for_track(
        24,
        team_evidence={},
        team_hints=hints,
    )

    assert team_id == "team_b"
    assert confidence == 0.94
    assert source == "fresh_on_court_team_hint"


def main() -> None:
    test_fresh_hint_requires_explicit_on_court_support()
    test_fresh_hint_keeps_existing_team_thresholds()
    test_public_diagnostics_can_fill_unknown_without_mutating_internal_team()
    test_shot_actor_lookup_can_use_hint_without_entering_possession_evidence()
    print("p2_team_attribution_hint_test: PASS")


if __name__ == "__main__":
    main()
