from __future__ import annotations

import statistics
from typing import Any

import cv2
import numpy as np

from core.config import Settings, settings
from core.court_state import CourtState
from services.vision.court_geometry import project_image_point_to_court


class CourtStateEstimatorV1:
    """Build P1 CourtState and conservative on-court eligibility.

    Two evidence modes are intentionally explicit:

    - ``calibrated``: a valid non-identity homography projects player/referee feet into
      FIBA court metres.
    - ``image_space_fallback``: no metric geometry is claimed; a court-colour surface
      mask plus wide-view/player-count evidence is used only for conservative QA and
      metric scoping.

    UNKNOWN remains valid. In particular, image-space fallback does not invent court
    coordinates and ambiguous entity eligibility remains ``None`` rather than False.
    """

    def __init__(
        self,
        *,
        config: Settings | None = None,
        on_court_margin_m: float = 0.75,
        local_court_accept_ratio: float = 0.18,
        local_court_reject_ratio: float = 0.05,
        min_analysis_players: int = 4,
        max_wide_view_player_height_ratio: float = 0.45,
    ) -> None:
        self.config = config or settings
        self.on_court_margin_m = max(0.0, float(on_court_margin_m))
        self.local_court_accept_ratio = max(0.0, min(float(local_court_accept_ratio), 1.0))
        self.local_court_reject_ratio = max(
            0.0,
            min(float(local_court_reject_ratio), self.local_court_accept_ratio),
        )
        self.min_analysis_players = max(1, int(min_analysis_players))
        self.max_wide_view_player_height_ratio = max(
            0.05,
            min(float(max_wide_view_player_height_ratio), 1.0),
        )
        self.last_state: CourtState | None = None
        self.last_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _court_mask(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([5, 50, 80], dtype=np.uint8)
        upper = np.array([25, 250, 255], dtype=np.uint8)
        return cv2.inRange(hsv, lower, upper)

    @staticmethod
    def _camera_diagnostics(frame_state: Any) -> dict[str, Any]:
        return dict((getattr(frame_state, "diagnostics", {}) or {}).get("camera", {}) or {})

    @classmethod
    def _calibrated(cls, frame_state: Any) -> bool:
        camera = cls._camera_diagnostics(frame_state)
        H = getattr(frame_state, "homography_matrix", None)
        return bool(
            H is not None
            and bool(camera.get("valid", False))
            and not bool(camera.get("homography_is_identity", True))
        )

    @staticmethod
    def _safe_box(entity: dict[str, Any]) -> tuple[float, float, float, float] | None:
        raw = entity.get("box", entity.get("bbox"))
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            return None
        try:
            x1, y1, x2, y2 = (float(raw[index]) for index in range(4))
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _foot_point(box: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, _, x2, y2 = box
        return (x1 + x2) / 2.0, y2

    def _inside_fiba(self, court_xy: tuple[float, float]) -> bool:
        x_m, y_m = court_xy
        margin = self.on_court_margin_m
        return bool(
            -margin <= x_m <= float(self.config.court_length) + margin
            and -margin <= y_m <= float(self.config.court_width) + margin
        )

    @staticmethod
    def _local_court_ratio(
        mask: np.ndarray,
        box: tuple[float, float, float, float],
    ) -> float | None:
        height, width = mask.shape[:2]
        x1, y1, x2, y2 = box
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        foot_x = (x1 + x2) / 2.0

        patch_x1 = max(0, int(round(foot_x - box_w * 0.40)))
        patch_x2 = min(width, int(round(foot_x + box_w * 0.40)))
        patch_y1 = max(0, int(round(y2 - box_h * 0.06)))
        patch_y2 = min(height, int(round(y2 + box_h * 0.12)))
        if patch_x2 <= patch_x1 or patch_y2 <= patch_y1:
            return None
        patch = mask[patch_y1:patch_y2, patch_x1:patch_x2]
        if patch.size == 0:
            return None
        return float(cv2.countNonZero(patch)) / float(patch.size)

    def _entity_eligibility(
        self,
        frame_state: Any,
        entity: dict[str, Any],
        mask: np.ndarray,
    ) -> tuple[bool | None, str, float | None]:
        box = self._safe_box(entity)
        if box is None:
            return None, "invalid_bbox", None

        if self._calibrated(frame_state):
            foot = self._foot_point(box)
            court_xy = project_image_point_to_court(
                foot,
                getattr(frame_state, "homography_matrix", None),
            )
            if court_xy is None:
                return None, "calibrated_projection_failed", None
            entity["court_xy"] = [round(float(court_xy[0]), 4), round(float(court_xy[1]), 4)]
            entity["court_projection_source"] = "homography_ground_plane"
            return self._inside_fiba(court_xy), "calibrated_fiba_bounds", None

        local_ratio = self._local_court_ratio(mask, box)
        if local_ratio is None:
            return None, "image_space_patch_unavailable", None
        if local_ratio >= self.local_court_accept_ratio:
            return True, "image_space_court_surface", local_ratio
        if local_ratio <= self.local_court_reject_ratio:
            return False, "image_space_off_court_surface", local_ratio
        return None, "image_space_ambiguous_surface", local_ratio

    def enrich_entities(self, frame_state: Any) -> Any:
        frame = getattr(frame_state, "frame", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return frame_state

        mask = self._court_mask(frame)
        for collection_name in ("players", "referees"):
            for entity in list(getattr(frame_state, collection_name, []) or []):
                if not isinstance(entity, dict):
                    continue
                active, source, local_ratio = self._entity_eligibility(frame_state, entity, mask)
                entity["active_on_court"] = active
                entity["court_eligibility_source"] = source
                entity["identity_eligible"] = active is not False
                if local_ratio is not None:
                    entity["local_court_surface_ratio"] = round(float(local_ratio), 6)
        return frame_state

    @staticmethod
    def _polygon(frame_state: Any) -> tuple[tuple[float, float], ...] | None:
        polygon = getattr(frame_state, "camera_polygon", None)
        if polygon is None:
            polygon = getattr(frame_state, "court_polygon", None)
        if polygon is None:
            return None
        try:
            values = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        except (TypeError, ValueError):
            return None
        if len(values) < 3 or not np.isfinite(values).all():
            return None
        return tuple((float(point[0]), float(point[1])) for point in values)

    @staticmethod
    def _height_ratios(frame_state: Any) -> list[float]:
        frame = getattr(frame_state, "frame", None)
        if frame is None or getattr(frame, "shape", None) is None:
            return []
        frame_h = max(float(frame.shape[0]), 1.0)
        ratios: list[float] = []
        for player in list(getattr(frame_state, "players", []) or []):
            if not isinstance(player, dict):
                continue
            raw = player.get("box", player.get("bbox"))
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                ratio = max(0.0, float(raw[3]) - float(raw[1])) / frame_h
            except (TypeError, ValueError):
                continue
            ratios.append(ratio)
        return ratios

    @staticmethod
    def _count_true(entities: list[dict[str, Any]]) -> int:
        return sum(1 for entity in entities if entity.get("active_on_court") is True)

    @staticmethod
    def _count_not_false(entities: list[dict[str, Any]]) -> int:
        return sum(1 for entity in entities if entity.get("active_on_court") is not False)

    def update(
        self,
        frame_state: Any,
        *,
        gameplay_diagnostics: dict[str, Any] | None = None,
    ) -> CourtState:
        self.enrich_entities(frame_state)
        gameplay = dict(gameplay_diagnostics or {})
        frame = getattr(frame_state, "frame", None)

        court_ratio = gameplay.get("court_ratio")
        if court_ratio is None and frame is not None and isinstance(frame, np.ndarray) and frame.size:
            mask = self._court_mask(frame)
            h, w = mask.shape[:2]
            roi = mask[int(h * 0.2):int(h * 0.8), int(w * 0.15):int(w * 0.85)]
            court_ratio = (
                float(cv2.countNonZero(roi)) / float(roi.size)
                if roi.size
                else 0.0
            )
        court_ratio_value = None if court_ratio is None else float(court_ratio)

        players = [
            item
            for item in list(getattr(frame_state, "players", []) or [])
            if isinstance(item, dict)
        ]
        referees = [
            item
            for item in list(getattr(frame_state, "referees", []) or [])
            if isinstance(item, dict)
        ]
        ratios = self._height_ratios(frame_state)
        median_height_ratio = statistics.median(ratios) if ratios else None
        wide_view = bool(
            len(ratios) >= 2
            and median_height_ratio is not None
            and median_height_ratio <= self.max_wide_view_player_height_ratio
        )

        geometry_valid = self._calibrated(frame_state)
        mode = "calibrated" if geometry_valid else "image_space_fallback"
        court_visible = bool(
            court_ratio_value is not None
            and court_ratio_value >= float(self.config.min_court_visibility)
        )

        on_court_players = self._count_true(players)
        plausible_players = self._count_not_false(players)
        on_court_referees = self._count_true(referees)

        if geometry_valid:
            enough_players = on_court_players >= self.min_analysis_players
        else:
            # In fallback mode UNKNOWN remains usable evidence. Only explicit off-court
            # rejections are removed from the plausible on-court population.
            enough_players = plausible_players >= self.min_analysis_players

        analysis_eligible = bool(court_visible and wide_view and enough_players)
        if not court_visible:
            reason = "court_not_visible"
        elif not wide_view:
            reason = "not_wide_basketball_view"
        elif not enough_players:
            reason = "insufficient_on_court_players"
        else:
            reason = "court_and_entities_consistent"

        camera = self._camera_diagnostics(frame_state)
        state = CourtState(
            frame_id=int(getattr(frame_state, "frame_id", 0)),
            timestamp_sec=(
                None
                if getattr(frame_state, "timestamp", None) is None
                else float(getattr(frame_state, "timestamp"))
            ),
            mode=mode,
            geometry_valid=geometry_valid,
            court_visible=court_visible,
            analysis_eligible=analysis_eligible,
            eligibility_reason=reason,
            court_surface_ratio=court_ratio_value,
            court_polygon_px=self._polygon(frame_state),
            homography_source=camera.get("source"),
            reprojection_rmse_px=camera.get("reprojection_rmse_px"),
            view_mode="wide" if wide_view else ("tight" if ratios else "unknown"),
            player_count=len(players),
            on_court_player_count=on_court_players,
            referee_count=len(referees),
            on_court_referee_count=on_court_referees,
            basket_evidence=[],
            provenance={
                "version": "court_state_v1",
                "detector_model": self.config.model,
                "detector_version": self.config.detector_version,
                "court_dimensions_m": [
                    float(self.config.court_length),
                    float(self.config.court_width),
                ],
                "calibrated_geometry_required_for_metric_coordinates": True,
                "image_space_fallback_claims_metric_coordinates": False,
                "min_analysis_players": self.min_analysis_players,
                "max_wide_view_player_height_ratio": self.max_wide_view_player_height_ratio,
                "local_court_accept_ratio": self.local_court_accept_ratio,
                "local_court_reject_ratio": self.local_court_reject_ratio,
                "basket_evidence_status": "not_evaluated_in_current_detection_only_path",
                "p2_event_semantics": False,
            },
        )
        self.last_state = state
        self.last_diagnostics = state.as_dict()
        return state

    def metadata(self) -> dict[str, Any]:
        return {
            "version": "court_state_v1",
            "modes": ["calibrated", "image_space_fallback"],
            "court_length_m": float(self.config.court_length),
            "court_width_m": float(self.config.court_width),
            "min_analysis_players": self.min_analysis_players,
            "max_wide_view_player_height_ratio": self.max_wide_view_player_height_ratio,
            "p2_event_semantics": False,
        }
