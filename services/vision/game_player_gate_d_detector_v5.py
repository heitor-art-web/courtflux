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

    Gate-D continuity diagnostics intentionally follow ``CourtState.gameplay_present``
    rather than the stricter metric-only ``analysis_eligible`` flag. This is validation
    scope only: canonical Identity-v2 processing still runs before this hook with its
    normal conservative eligibility policy. The wider scope lets Gate D score visually
    resolvable players through tight/limited gameplay views without changing P1/P2 state.
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

    @staticmethod
    def _gate_d_widen_continuity_scope(frame_state: Any) -> bool:
        court_state = getattr(frame_state, "court_state", None)
        return bool(
            court_state is not None
            and bool(getattr(court_state, "gameplay_present", False))
            and not bool(getattr(court_state, "analysis_eligible", False))
        )

    def _observe_game_players(self, frame_state: Any) -> None:
        # ``GateDDetectorV4._observe_game_players`` reuses the canonical
        # ``_identity_eligible_players`` helper, whose scope is intentionally restricted
        # to ``analysis_eligible`` frames. For Gate-D continuity validation only, widen
        # that one post-process observation call to high-recall gameplay presence. The
        # original CourtState value is restored immediately, so no canonical state or
        # upstream identity/team processing is mutated.
        court_state = getattr(frame_state, "court_state", None)
        widen_scope = self._gate_d_widen_continuity_scope(frame_state)
        original_analysis_eligible = (
            bool(getattr(court_state, "analysis_eligible", False))
            if court_state is not None
            else False
        )
        if widen_scope and court_state is not None:
            court_state.analysis_eligible = True
        try:
            super()._observe_game_players(frame_state)
        finally:
            if widen_scope and court_state is not None:
                court_state.analysis_eligible = original_analysis_eligible

        detector = frame_state.diagnostics.setdefault("detector", {})
        runtime = detector.setdefault("game_player_runtime", {})
        runtime["version"] = "game_player_resolver_v5_gate_d"
        runtime["deferred_ambiguous_reassociation"] = True
        runtime["deferred_long_gap_reappearance"] = False
        runtime["bounded_reappearance_probe"] = False
        runtime["continuity_scope"] = "gameplay_present"
        runtime["limited_gameplay_scope_widened"] = bool(widen_scope)
