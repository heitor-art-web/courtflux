from __future__ import annotations

import os
from typing import Any

from services.p2.airborne_classifier_two_stage_v1 import AirborneClassifierV1
from services.p2.trained_airborne_live_control_v1 import (
    LiveControlP2EngineV1 as _SingleStageLiveControlP2EngineV1,
)


class LiveControlP2EngineV1(_SingleStageLiveControlP2EngineV1):
    """Canonical resolver using the two-stage AirborneClassifierV1 artifact."""

    LIVE_CONTROL_POLICY = (
        "causal_live_resolver_v5;two_stage_airborne_classifier;"
        "stage1_airborne_vs_continuity;stage2_pass_vs_shot;"
        "strong_shot_signals_precede_classifier;events_unchanged"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        explicit_model_path = kwargs.pop("airborne_model_path", None)
        env_model_path = os.environ.pop("P2_AIRBORNE_MODEL", None)
        try:
            # Prevent the inherited single-stage classifier from trying to load the new
            # staged artifact. It is replaced immediately after base initialization.
            super().__init__(*args, airborne_model_path=None, **kwargs)
        finally:
            if env_model_path is not None:
                os.environ["P2_AIRBORNE_MODEL"] = env_model_path

        model_path = explicit_model_path or env_model_path
        self.airborne_classifier = AirborneClassifierV1(model_path)
