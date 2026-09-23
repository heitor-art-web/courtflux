from __future__ import annotations

from typing import Any

from services.vision.game_player_gate_d_detector_v2 import PerformanceDetector as GateDDetectorV2
from services.vision.game_player_resolver_gate_d_v3 import GateDGamePlayerResolverV3


class PerformanceDetector(GateDDetectorV2):
    """Gate-D detector using the v3 global visual-shift preflight."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.game_player_resolver = GateDGamePlayerResolverV3()
        print(
            "🧍 Gate D global visual-shift preflight: v3 raw+coarse aligned",
            flush=True,
        )
