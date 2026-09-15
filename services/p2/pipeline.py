from __future__ import annotations

import os
from typing import Any

from services.pipeline import Pipeline
from services.p2.engine_v1 import P2EngineV1


class P2Pipeline(Pipeline):
    """Current-main Pipeline with the rebuilt P2 stage inserted after canonical P1 state.

    The base ``Pipeline`` remains untouched. This adapter preserves its detection-only,
    legacy analytics, persistence, renderer, and PR #43 P1 behavior while adding P2
    basketball-understanding events at the P1 -> P2 boundary.
    """

    def __init__(self, *args: Any, p2_enabled: bool | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if p2_enabled is None:
            raw = os.getenv("BASKETBALL_P2_ENABLED", "0").strip().lower()
            p2_enabled = raw not in {"", "0", "false", "no", "off"}
        self.p2_enabled = bool(p2_enabled)
        self.p2_engine: P2EngineV1 | None = None

    def _attach_canonical_p1_states(self, frame_state: Any):
        frame_state = super()._attach_canonical_p1_states(frame_state)

        if not self.p2_enabled:
            frame_state.diagnostics.setdefault(
                "p2",
                {
                    "enabled": False,
                    "reason": "set BASKETBALL_P2_ENABLED=1 or pass p2_enabled=True",
                },
            )
            return frame_state

        if self.p2_engine is None:
            self.p2_engine = P2EngineV1(
                fps=float(getattr(frame_state, "fps", 30.0) or 30.0)
            )

        frame_state = self.p2_engine.process(frame_state)
        frame_state.diagnostics.setdefault("p2", {})["enabled"] = True
        frame_state.diagnostics["p2"]["integration"] = "P2Pipeline.after_canonical_p1_state"
        return frame_state
