from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CourtState:
    """Canonical P1 court/view state for one video frame.

    CourtState separates two questions that must not be conflated:

    - ``gameplay_present`` / ``gameplay_status``: whether detector-level basketball
      evidence is sufficient to preserve tracking/Identity/P2 continuity.
    - ``analysis_eligible``: whether the frame is a conservative wide/court-visible view
      suitable for metric-scoped P1 analysis.

    CourtState remains perception/state evidence only. It deliberately does not encode
    possession, shot outcome, made/missed baskets, or other P2 event truth.
    """

    frame_id: int
    timestamp_sec: float | None
    mode: str = "unavailable"
    geometry_valid: bool = False
    court_visible: bool = False
    analysis_eligible: bool = False
    eligibility_reason: str = "unavailable"
    gameplay_present: bool | None = None
    gameplay_status: str = "unknown"
    gameplay_reason: str = "unavailable"
    court_surface_ratio: float | None = None
    court_polygon_px: tuple[tuple[float, float], ...] | None = None
    homography_source: str | None = None
    reprojection_rmse_px: float | None = None
    view_mode: str = "unknown"
    player_count: int = 0
    on_court_player_count: int = 0
    referee_count: int = 0
    on_court_referee_count: int = 0
    basket_evidence: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_id = int(self.frame_id)
        if self.timestamp_sec is not None:
            self.timestamp_sec = float(self.timestamp_sec)
        if self.court_surface_ratio is not None:
            self.court_surface_ratio = max(0.0, min(float(self.court_surface_ratio), 1.0))
        if self.reprojection_rmse_px is not None:
            self.reprojection_rmse_px = max(0.0, float(self.reprojection_rmse_px))
        self.player_count = max(0, int(self.player_count))
        self.on_court_player_count = max(0, int(self.on_court_player_count))
        self.referee_count = max(0, int(self.referee_count))
        self.on_court_referee_count = max(0, int(self.on_court_referee_count))

        # Derive high-recall gameplay presence from richer post-detector evidence while
        # keeping analysis_eligible conservative. Two detected players are sufficient to
        # preserve temporal continuity even when court colour/wide-view checks abstain.
        if self.gameplay_present is None:
            if self.analysis_eligible:
                self.gameplay_present = True
                self.gameplay_status = "gameplay_analysis"
                self.gameplay_reason = "analysis_eligible"
            elif self.player_count >= 2:
                self.gameplay_present = True
                self.gameplay_status = "gameplay_limited"
                self.gameplay_reason = "multiple_player_evidence"
            elif self.player_count >= 1 and (self.court_visible or bool(self.basket_evidence)):
                self.gameplay_present = True
                self.gameplay_status = "gameplay_limited"
                self.gameplay_reason = "player_plus_basketball_context"
            else:
                self.gameplay_present = False
                self.gameplay_status = "non_gameplay"
                self.gameplay_reason = "insufficient_post_detector_basketball_evidence"
        else:
            self.gameplay_present = bool(self.gameplay_present)
            if self.gameplay_status == "unknown":
                if self.analysis_eligible and self.gameplay_present:
                    self.gameplay_status = "gameplay_analysis"
                elif self.gameplay_present:
                    self.gameplay_status = "gameplay_limited"
                else:
                    self.gameplay_status = "non_gameplay"
            if self.gameplay_reason == "unavailable":
                self.gameplay_reason = "explicit_gameplay_presence"

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_sec": self.timestamp_sec,
            "mode": self.mode,
            "geometry_valid": bool(self.geometry_valid),
            "court_visible": bool(self.court_visible),
            "analysis_eligible": bool(self.analysis_eligible),
            "eligibility_reason": self.eligibility_reason,
            "gameplay_present": bool(self.gameplay_present),
            "gameplay_status": self.gameplay_status,
            "gameplay_reason": self.gameplay_reason,
            "court_surface_ratio": self.court_surface_ratio,
            "court_polygon_px": (
                [list(point) for point in self.court_polygon_px]
                if self.court_polygon_px is not None
                else None
            ),
            "homography_source": self.homography_source,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "view_mode": self.view_mode,
            "player_count": self.player_count,
            "on_court_player_count": self.on_court_player_count,
            "referee_count": self.referee_count,
            "on_court_referee_count": self.on_court_referee_count,
            "basket_evidence": [dict(item) for item in self.basket_evidence],
            "provenance": dict(self.provenance),
        }
