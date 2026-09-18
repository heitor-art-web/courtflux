from typing import Any

from . import engine_v1 as _engine_v1
from .possession_tracker_v1 import PossessionTrackerV1
from .shot_event_tracker_v1 import ShotEventTrackerV1
from .two_stage_airborne_live_control_v1 import LiveControlP2EngineV1

# Keep a single canonical event engine. The live-control resolver subclasses the existing
# P2EngineV1 event path and only changes the published control-state semantics afterwards.
# AirborneClassifierV1 is optional: without a trained staged artifact behavior falls back to
# AIRBORNE-v2; with a model it resolves AIRBORNE-vs-CONTINUITY first, then PASS-vs-SHOT.
_engine_v1.P2EngineV1 = LiveControlP2EngineV1
P2EngineV1 = LiveControlP2EngineV1

__all__ = ["P2EngineV1", "P2Pipeline", "PossessionTrackerV1", "ShotEventTrackerV1"]


def __getattr__(name: str) -> Any:
    if name == "P2Pipeline":
        from .pipeline import P2Pipeline

        return P2Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
