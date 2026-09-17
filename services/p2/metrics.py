from __future__ import annotations

from typing import Any

from services.p2.metrics_base_v5 import P2MetricsCollector as _BaseP2MetricsCollector


class P2MetricsCollector(_BaseP2MetricsCollector):
    """P2 metrics with calibration diagnostics and provisional resolved control state.

    ``control_state`` remains the live online P2 decision. ``resolved_control_state`` is
    an additive, retrospective evaluation channel: when the shot tracker later activates,
    a directly preceding generic flight segment from the same temporal-reset epoch may be
    relabeled as ``shot_flight``. This never mutates live P2 state or the live metrics.
    """

    RESOLVED_CONTROL_POLICY = (
        "live_state_immutable;late_shot_activation_may_backfill_preceding_generic_flight;"
        "receiver_evidence_abstains;resolved_state_is_provisional"
    )

    def __init__(self) -> None:
        super().__init__()
        self._previous_shot_active = False

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

        shot_active = bool(row.get("shot_active", False))
        if shot_active and not self._previous_shot_active:
            self._backfill_resolved_shot_flight(len(self.state_rows) - 1)
        self._previous_shot_active = shot_active
