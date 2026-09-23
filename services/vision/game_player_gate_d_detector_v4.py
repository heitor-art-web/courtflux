from __future__ import annotations

from typing import Any

from services.vision.game_player_gate_d_detector_v3 import PerformanceDetector as GateDDetectorV3
from services.vision.game_player_resolver_gate_d_v4 import GateDGamePlayerResolverV4


class PerformanceDetector(GateDDetectorV3):
    """Gate-D detector using temporal-cohort visual-shift suppression."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.game_player_resolver = GateDGamePlayerResolverV4()
        print(
            "🧍 Gate D global visual-shift preflight: v4 temporal cohort 30f",
            flush=True,
        )
