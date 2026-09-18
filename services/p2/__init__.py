from typing import Any

from . import engine_v1 as _engine_v1
from .live_control_resolver_v1 import LiveControlP2EngineV1
from .possession_tracker_v1 import PossessionTrackerV1
from .shot_event_tracker_v1 import ShotEventTrackerV1

# Keep a single canonical event engine. The AIRBORNE v2 live-control resolver subclasses
# the existing P2EngineV1 event path and only resolves published control-state semantics
# afterwards. The raw-receiver experiment remains in the repository for auditability but
# is intentionally not part of the canonical runtime after Gate-A regression testing.
_engine_v1.P2EngineV1 = LiveControlP2EngineV1
P2EngineV1 = LiveControlP2EngineV1

__all__ = ["P2EngineV1", "P2Pipeline", "PossessionTrackerV1", "ShotEventTrackerV1"]


def __getattr__(name: str) -> Any:
    if name == "P2Pipeline":
        from .pipeline import P2Pipeline

        return P2Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
