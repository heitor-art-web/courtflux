from __future__ import annotations

import os
from typing import Any

from services.p2.engine_v6_batch import P2EngineV6Batch
from services.pipeline import Pipeline
from services.vision.court_state_gameplay_v3 import CourtStateEstimatorV3


class P2Pipeline(Pipeline):
    """Compatibility wrapper that forces the canonical Pipeline's P2 stage on.

    The V6 validation branch keeps all batch changes behind this P2-only wrapper:
    CourtState gameplay policy V3 plus the V6 engine coordinator. The base Pipeline and
    non-P2 runtime remain unchanged.
    """

    def __init__(self, *args: Any, p2_enabled: bool | None = None, **kwargs: Any) -> None:
        if p2_enabled is None:
            p2_enabled = True
        if p2_enabled:
            os.environ["BASKETBALL_P2_ENABLED"] = "1"
        else:
            os.environ["BASKETBALL_P2_ENABLED"] = "0"
        super().__init__(*args, **kwargs)
        gameplay_estimator = CourtStateEstimatorV3()
        self.court_state_estimator = gameplay_estimator
        if hasattr(self.detector, "identity_court_estimator"):
            self.detector.identity_court_estimator = gameplay_estimator

    def process(self, frame_state: Any):
        # The base Pipeline instantiates its frozen P2EngineV1 only when p2_engine is None.
        # Preinstall the V6 coordinator here so the isolated P2 validation path uses the
        # batch without changing the global Pipeline contract.
        if self.p2_enabled and self.p2_engine is None:
            self.p2_engine = P2EngineV6Batch(
                fps=float(getattr(frame_state, "fps", 30.0) or 30.0)
            )
        return super().process(frame_state)
