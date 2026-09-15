from typing import Any

from .engine_v1 import P2EngineV1
from .possession_tracker_v1 import PossessionTrackerV1
from .shot_event_tracker_v1 import ShotEventTrackerV1

__all__ = ["P2EngineV1", "P2Pipeline", "PossessionTrackerV1", "ShotEventTrackerV1"]


def __getattr__(name: str) -> Any:
    if name == "P2Pipeline":
        from .pipeline import P2Pipeline

        return P2Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
