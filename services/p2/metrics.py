from __future__ import annotations

import math
from typing import Any

from services.p2.metrics_base_v5 import P2MetricsCollector as _BaseP2MetricsCollector


class P2MetricsCollector(_BaseP2MetricsCollector):
    """P2 metrics with calibration diagnostics and provisional challenger channels.

    ``control_state`` remains the live online P2 decision. ``resolved_control_state`` is
    an additive, retrospective evaluation channel: when the shot tracker later activates,
    a directly preceding generic flight segment from the same temporal-reset epoch may be
    relabeled as ``shot_flight``. ``soft_acquisition_challenger_state`` is another additive
    evaluation-only channel for a conservative ownerless/no-flight possession hypothesis.
    Neither channel mutates live P2 state or live metrics.

    Shot-preflight diagnostics are causal per-frame observations computed from the current
    FrameState only. They expose ball/rim geometry and short temporal trends for evaluating
    an online shot-preactivation challenger without changing shot/control decisions.
    """

    RESOLVED_CONTROL_POLICY = (
        "live_state_immutable;late_shot_activation_may_backfill_preceding_generic_flight;"
        "receiver_evidence_abstains;resolved_state_is_provisional"
    )
    SOFT_ACQUISITION_MIN_STREAK_FRAMES = 4
    SOFT_ACQUISITION_MIN_MEAN_SCORE = 0.35
    SOFT_ACQUISITION_POLICY = (
        "diagnostic_only;live_unknown;engine_eligible;owner_absent;thresholded_candidate_absent;"
        "flight_inactive;shot_inactive;same_raw_best_track>=4_frames;raw_best_mean_score>=0.35"
    )
    SHOT_PREFLIGHT_POLICY = (
        "diagnostic_only;causal_current_and_past_frames_only;nearest_rim_geometry;"
        "no_live_or_resolved_state_mutation"
    )

    def __init__(self) -> None:
        super().__init__()
        self._previous_shot_active = False
        self._soft_track_id: int | None = None
        self._soft_streak_frames = 0
        self._soft_score_sum = 0.0
        self._soft_score_peak = 0.0
        self._soft_last_frame: int | None = None
        self._preflight_last_frame: int | None = None
        self._preflight_last_distance_norm: float | None = None
        self._preflight_last_relative_y: float | None = None
        self._preflight_approach_streak = 0
        self._preflight_rising_streak = 0

    @staticmethod
    def _same_reset(a: dict[str, Any], b: dict[str, Any]) -> bool:
        try:
            return int(a.get("reset_count", 0) or 0) == int(b.get("reset_count", 0) or 0)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _flight_start_from_row(row: dict[str, Any]) -> int | None:
        if not bool(row.get("flight_active", False)):
            return None
        try:
            frame_id = int(row.get("frame_id", -1))
            flight_frames = max(1, int(row.get("flight_frames", 0) or 0))
        except (TypeError, ValueError):
            return None
        if frame_id < 0:
            return None
        return max(0, frame_id - flight_frames + 1)

    @staticmethod
    def _preflight_ball_position(frame_state: Any) -> tuple[float, float] | None:
        ball_state = getattr(frame_state, "ball_state", None)
        if ball_state is None or not bool(getattr(ball_state, "available", False)):
            return None
        position_fn = getattr(ball_state, "position", None)
        if not callable(position_fn):
            return None
        position = position_fn()
        if position is None:
            return None
        try:
            return float(position[0]), float(position[1])
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _preflight_direct_observed(frame_state: Any) -> bool:
        ball_state = getattr(frame_state, "ball_state", None)
        if ball_state is None:
            return False
        status = getattr(ball_state, "status", None)
        status_name = str(getattr(status, "name", status) or "").strip().lower()
        return bool(status_name == "observed" and P2MetricsCollector._preflight_ball_position(frame_state))

    @staticmethod
    def _preflight_rims(frame_state: Any) -> list[tuple[float, float, float, float]]:
        court_state = getattr(frame_state, "court_state", None)
        if court_state is None:
            return []
        result: list[tuple[float, float, float, float]] = []
        for evidence in list(getattr(court_state, "basket_evidence", None) or []):
            if not isinstance(evidence, dict) or str(evidence.get("kind") or "").lower() != "rim":
                continue
            raw = evidence.get("bbox_px", evidence.get("box", evidence.get("bbox")))
            if not isinstance(raw, (list, tuple)) or len(raw) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in raw[:4]]
            except (TypeError, ValueError):
                continue
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2, y2))
        return result

    def _update_shot_preflight(self, row: dict[str, Any], frame_state: Any) -> None:
        try:
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            frame_id = -1
        position = self._preflight_ball_position(frame_state)
        rims = self._preflight_rims(frame_state)
        direct_observed = self._preflight_direct_observed(frame_state)

        rim = None
        if position is not None and rims:
            bx, by = position
            rim = min(
                rims,
                key=lambda candidate: (
                    ((candidate[0] + candidate[2]) / 2.0 - bx) ** 2
                    + ((candidate[1] + candidate[3]) / 2.0 - by) ** 2
                ),
            )

        if frame_id < 0 or position is None or rim is None:
            self._preflight_last_frame = None
            self._preflight_last_distance_norm = None
            self._preflight_last_relative_y = None
            self._preflight_approach_streak = 0
            self._preflight_rising_streak = 0
            row.update(
                {
                    "shot_preflight_available": False,
                    "shot_preflight_direct_observed": direct_observed,
                    "shot_preflight_near_x": False,
                    "shot_preflight_above": False,
                    "shot_preflight_rim_band": False,
                    "shot_preflight_trajectory_start": False,
                    "shot_preflight_ball_to_rim_distance_norm": None,
                    "shot_preflight_horizontal_distance_rim_widths": None,
                    "shot_preflight_relative_y_rim_heights": None,
                    "shot_preflight_delta_distance_norm": None,
                    "shot_preflight_delta_relative_y": None,
                    "shot_preflight_approach_streak": 0,
                    "shot_preflight_rising_streak": 0,
                    "shot_preflight_policy": self.SHOT_PREFLIGHT_POLICY,
                }
            )
            return

        bx, by = position
        x1, y1, x2, y2 = rim
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        rim_x = (x1 + x2) / 2.0
        rim_y = (y1 + y2) / 2.0
        dx = bx - rim_x
        dy = by - rim_y
        horizontal_widths = abs(dx) / width
        relative_y = dy / height
        distance_norm = math.hypot(dx / width, dy / height)
        near_x = horizontal_widths <= 2.5
        above = relative_y < -0.20
        rim_band = bool((-1.3 <= relative_y <= 1.3) and near_x)
        trajectory_start = bool(above and near_x)

        consecutive = bool(
            direct_observed
            and self._preflight_last_frame is not None
            and frame_id == self._preflight_last_frame + 1
            and self._preflight_last_distance_norm is not None
            and self._preflight_last_relative_y is not None
        )
        delta_distance = None
        delta_relative_y = None
        if consecutive:
            delta_distance = distance_norm - float(self._preflight_last_distance_norm)
            delta_relative_y = relative_y - float(self._preflight_last_relative_y)
            self._preflight_approach_streak = (
                self._preflight_approach_streak + 1 if delta_distance < 0.0 else 0
            )
            self._preflight_rising_streak = (
                self._preflight_rising_streak + 1 if delta_relative_y < 0.0 else 0
            )
        else:
            self._preflight_approach_streak = 0
            self._preflight_rising_streak = 0

        if direct_observed:
            self._preflight_last_frame = frame_id
            self._preflight_last_distance_norm = distance_norm
            self._preflight_last_relative_y = relative_y
        else:
            self._preflight_last_frame = None
            self._preflight_last_distance_norm = None
            self._preflight_last_relative_y = None
            self._preflight_approach_streak = 0
            self._preflight_rising_streak = 0

        row.update(
            {
                "shot_preflight_available": True,
                "shot_preflight_direct_observed": direct_observed,
                "shot_preflight_near_x": near_x,
                "shot_preflight_above": above,
                "shot_preflight_rim_band": rim_band,
                "shot_preflight_trajectory_start": trajectory_start,
                "shot_preflight_ball_to_rim_distance_norm": round(distance_norm, 6),
                "shot_preflight_horizontal_distance_rim_widths": round(horizontal_widths, 6),
                "shot_preflight_relative_y_rim_heights": round(relative_y, 6),
                "shot_preflight_delta_distance_norm": (
                    None if delta_distance is None else round(delta_distance, 6)
                ),
                "shot_preflight_delta_relative_y": (
                    None if delta_relative_y is None else round(delta_relative_y, 6)
                ),
                "shot_preflight_approach_streak": int(self._preflight_approach_streak),
                "shot_preflight_rising_streak": int(self._preflight_rising_streak),
                "shot_preflight_policy": self.SHOT_PREFLIGHT_POLICY,
            }
        )

    def _reset_soft_acquisition(self) -> None:
        self._soft_track_id = None
        self._soft_streak_frames = 0
        self._soft_score_sum = 0.0
        self._soft_score_peak = 0.0
        self._soft_last_frame = None

    def _update_soft_acquisition(self, row: dict[str, Any]) -> None:
        try:
            frame_id = int(row.get("frame_id", -1))
        except (TypeError, ValueError):
            frame_id = -1
        raw_track = row.get("raw_best_track_id")
        try:
            raw_track_id = None if raw_track is None else int(raw_track)
            raw_score = float(row.get("raw_best_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            raw_track_id = None
            raw_score = 0.0

        eligible = bool(
            frame_id >= 0
            and bool(row.get("engine_eligible", False))
            and str(row.get("live_control_state") or row.get("control_state") or "unknown")
            == "unknown"
            and row.get("current_track_id") is None
            and row.get("candidate_track_id") is None
            and not bool(row.get("flight_active", False))
            and not bool(row.get("shot_active", False))
            and raw_track_id is not None
        )

        if not eligible:
            self._reset_soft_acquisition()
            row.update(
                {
                    "soft_acquisition_eligible": False,
                    "soft_acquisition_track_id": None,
                    "soft_acquisition_streak_frames": 0,
                    "soft_acquisition_mean_score": 0.0,
                    "soft_acquisition_peak_score": 0.0,
                    "soft_acquisition_triggered": False,
                    "soft_acquisition_challenger_state": str(
                        row.get("live_control_state") or row.get("control_state") or "unknown"
                    ),
                    "soft_acquisition_challenger_source": "live",
                    "soft_acquisition_policy": self.SOFT_ACQUISITION_POLICY,
                }
            )
            return

        consecutive_same_track = bool(
            self._soft_track_id is not None
            and self._soft_track_id == raw_track_id
            and self._soft_last_frame is not None
            and frame_id == self._soft_last_frame + 1
        )
        if consecutive_same_track:
            self._soft_streak_frames += 1
            self._soft_score_sum += raw_score
            self._soft_score_peak = max(self._soft_score_peak, raw_score)
        else:
            self._soft_track_id = raw_track_id
            self._soft_streak_frames = 1
            self._soft_score_sum = raw_score
            self._soft_score_peak = raw_score
        self._soft_last_frame = frame_id

        mean_score = self._soft_score_sum / max(self._soft_streak_frames, 1)
        triggered = bool(
            self._soft_streak_frames >= self.SOFT_ACQUISITION_MIN_STREAK_FRAMES
            and mean_score >= self.SOFT_ACQUISITION_MIN_MEAN_SCORE
        )
        row.update(
            {
                "soft_acquisition_eligible": True,
                "soft_acquisition_track_id": self._soft_track_id,
                "soft_acquisition_streak_frames": int(self._soft_streak_frames),
                "soft_acquisition_mean_score": round(float(mean_score), 6),
                "soft_acquisition_peak_score": round(float(self._soft_score_peak), 6),
                "soft_acquisition_triggered": triggered,
                "soft_acquisition_challenger_state": (
                    "possessed"
                    if triggered
                    else str(row.get("live_control_state") or row.get("control_state") or "unknown")
                ),
                "soft_acquisition_challenger_source": (
                    "temporal_soft_acquisition" if triggered else "live"
                ),
                "soft_acquisition_policy": self.SOFT_ACQUISITION_POLICY,
            }
        )

    def _backfill_resolved_shot_flight(self, activation_index: int) -> None:
        if activation_index < 0 or activation_index >= len(self.state_rows):
            return
        activation = self.state_rows[activation_index]
        try:
            activation_frame = int(activation.get("frame_id", -1))
        except (TypeError, ValueError):
            return
        if activation_frame < 0:
            return

        last_flight_index: int | None = None
        for probe in range(activation_index - 1, -1, -1):
            candidate = self.state_rows[probe]
            if not self._same_reset(candidate, activation):
                break
            if bool(candidate.get("flight_active", False)):
                last_flight_index = probe
                break
        if last_flight_index is None:
            return

        last_flight = self.state_rows[last_flight_index]
        flight_start = self._flight_start_from_row(last_flight)
        if flight_start is None:
            return
        try:
            last_flight_frame = int(last_flight.get("frame_id", -1))
        except (TypeError, ValueError):
            return

        receiver_evidence = False
        for candidate in self.state_rows:
            try:
                frame_id = int(candidate.get("frame_id", -1))
            except (TypeError, ValueError):
                continue
            if frame_id < flight_start or frame_id > last_flight_frame:
                continue
            if not self._same_reset(candidate, activation):
                continue
            if bool(candidate.get("pass_flight_receiver_evidence", False)):
                receiver_evidence = True
                break

        activation["resolved_shot_backfill_flight_start_frame"] = flight_start
        activation["resolved_shot_backfill_last_flight_frame"] = last_flight_frame
        activation["resolved_shot_backfill_receiver_evidence"] = receiver_evidence
        if receiver_evidence:
            activation["resolved_shot_backfill_applied"] = False
            return

        relabeled = 0
        for candidate in self.state_rows:
            try:
                frame_id = int(candidate.get("frame_id", -1))
            except (TypeError, ValueError):
                continue
            if frame_id < flight_start or frame_id > activation_frame:
                continue
            if not self._same_reset(candidate, activation):
                continue
            candidate["resolved_control_state"] = "shot_flight"
            candidate["resolved_control_source"] = "late_shot_activation_backfill"
            candidate["resolved_control_provisional"] = True
            candidate["resolved_shot_activation_frame"] = activation_frame
            relabeled += 1

        activation["resolved_shot_backfill_applied"] = relabeled > 0
        activation["resolved_shot_backfill_relabeled_frames"] = relabeled

    def observe(self, frame_state: Any) -> None:
        super().observe(frame_state)
        if not self.state_rows:
            return

        frame_id = int(getattr(frame_state, "frame_id", 0))
        row = self.state_rows[-1]
        if int(row.get("frame_id", -1)) != frame_id:
            return

        diagnostics = dict(getattr(frame_state, "diagnostics", {}) or {})
        p2 = dict(diagnostics.get("p2", {}) or {})
        possession = dict(p2.get("possession", {}) or {})
        flight = dict(possession.get("flight", {}) or {})

        row.update(
            {
                "candidate_track_id": possession.get("candidate_track_id"),
                "candidate_frames": possession.get("candidate_frames"),
                "candidate_score": possession.get("candidate_score"),
                "candidate_peak_score": possession.get("candidate_peak_score"),
                "candidate_threshold": possession.get("candidate_threshold"),
                "candidate_required_frames": possession.get("candidate_required_frames"),
                "raw_best_track_id": possession.get("raw_best_track_id"),
                "raw_best_score": possession.get("raw_best_score"),
                "raw_best_above_threshold": bool(
                    possession.get("raw_best_above_threshold", False)
                ),
                "raw_best_distinct_from_flight_origin": bool(
                    possession.get("raw_best_distinct_from_flight_origin", False)
                ),
                "pass_flight_receiver_evidence": bool(
                    possession.get("pass_flight_receiver_evidence", False)
                ),
                "flight_origin_track_id": flight.get("origin_track_id"),
                "flight_direct_observations": flight.get("flight_direct_observations"),
                "flight_predicted_observations": flight.get("flight_predicted_observations"),
                "flight_displacement_px": flight.get("flight_displacement_px"),
                "flight_displacement_player_heights": flight.get(
                    "flight_displacement_player_heights"
                ),
                "flight_classification_policy": possession.get(
                    "flight_classification_policy"
                ),
            }
        )

        live_state = str(row.get("control_state") or "unknown")
        row["live_control_state"] = live_state
        row["resolved_control_state"] = live_state
        row["resolved_control_source"] = "live"
        row["resolved_control_provisional"] = False
        row["resolved_control_policy"] = self.RESOLVED_CONTROL_POLICY

        self._update_shot_preflight(row, frame_state)
        self._update_soft_acquisition(row)

        shot_active = bool(row.get("shot_active", False))
        if shot_active and not self._previous_shot_active:
            self._backfill_resolved_shot_flight(len(self.state_rows) - 1)
        self._previous_shot_active = shot_active
