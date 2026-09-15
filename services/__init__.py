"""Service package exports.

Keep top-level imports lazy so lightweight subpackages (for example ``services.p2``)
can be tested without importing OpenCV or constructing the full video runtime.
"""

from typing import Any

__all__ = ["FrameLoader", "Pipeline", "VideoSession"]


def __getattr__(name: str) -> Any:
    if name == "FrameLoader":
        from .frame_loader import FrameLoader

        return FrameLoader
    if name == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline
    if name == "VideoSession":
        from .video_session import VideoSession

        return VideoSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
