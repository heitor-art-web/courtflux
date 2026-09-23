from __future__ import annotations

from typing import Any

from services.vision.game_player_gate_d_detector_v4 import PerformanceDetector as GateDDetectorV4
from services.vision.game_player_resolver_gate_d_v5 import (
    GateDDeferredAssociationConfig,
    GateDGamePlayerResolverV5,
)


class PerformanceDetector(GateDDetectorV4):
    """Gate-D detector using v5 persistent ambiguity recovery.

    The bounded long-gap reappearance experiment remains available in the resolver for
    synthetic coverage, but is disabled in the development runtime after V3 showed that
    it delayed many legitimate new identities without producing a safe reappearance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Keep the reappearance ceiling equal to the canonical association window so the
        # long-gap path has no runtime-only window. Persistent within-window ambiguity
        # recovery remains enabled.
        self.game_player_resolver = GateDGamePlayerResolverV5(
            deferred_association_config=GateDDeferredAssociationConfig(
                max_reappearance_gap_frames=180,
            )
        )
        print(
            "🧍 Gate D identity resolver: v5 deferred ambiguity; long-gap reappearance disabled",
            flush=True,
        )

    def _observe_game_players(self, frame_state: Any) -> None:
        super()._observe_game_players(frame_state)
        detector = frame_state.diagnostics.setdefault("detector", {})
        runtime = detector.setdefault("game_player_runtime", {})
        runtime["version"] = "game_player_resolver_v5_gate_d"
        runtime["deferred_ambiguous_reassociation"] = True
        runtime["deferred_long_gap_reappearance"] = False
        runtime["bounded_reappearance_probe"] = False
