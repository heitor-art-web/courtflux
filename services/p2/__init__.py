from typing import Any

from . import engine_v1 as _engine_v1
from .live_control_resolver_v1 import LiveControlP2EngineV1
from .possession_tracker_v1 import PossessionTrackerV1
from .shot_event_tracker_v1 import ShotEventTrackerV1

# Keep a single canonical event engine. The live-control class subclasses the existing
# P2EngineV1 and calls its event path exactly once, then resolves only the published
# control-state semantics. Rebind the submodule export so direct imports and Pipeline use
# the same canonical class.
_engine_v1.P2EngineV1 = LiveControlP2EngineV1
P2EngineV1 = LiveControlP2EngineV1

__all__ = ["P2EngineV1", "P2Pipeline", "PossessionTrackerV1", "ShotEventTrackerV1"]


def __getattr__(name: str) -> Any:
    if name == "P2Pipeline":
        from .pipeline import P2Pipeline

        return P2Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
