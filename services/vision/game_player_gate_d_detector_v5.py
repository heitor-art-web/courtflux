from __future__ import annotations

from typing import Any

from services.vision.game_player_gate_d_detector_v4 import PerformanceDetector as GateDDetectorV4
from services.vision.game_player_resolver_gate_d_v5 import GateDGamePlayerResolverV5


class PerformanceDetector(GateDDetectorV4):
    """Gate-D detector using v5 persistent ambiguity and reappearance recovery."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.game_player_resolver = GateDGamePlayerResolverV5()
        print(
            "🧍 Gate D identity resolver: v5 deferred ambiguity + bounded reappearance probe",
            flush=True,
        )

    def _observe_game_players(self, frame_state: Any) -> None:
        super()._observe_game_players(frame_state)
        detector = frame_state.diagnostics.setdefault("detector", {})
        runtime = detector.setdefault("game_player_runtime", {})
        runtime["version"] = "game_player_resolver_v5_gate_d"
        runtime["deferred_ambiguous_reassociation"] = True
        runtime["deferred_long_gap_reappearance"] = True
        runtime["bounded_reappearance_probe"] = True
