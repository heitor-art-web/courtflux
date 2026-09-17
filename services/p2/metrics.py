from __future__ import annotations

from typing import Any

from services.p2.metrics_base_v5 import P2MetricsCollector as _BaseP2MetricsCollector


class P2MetricsCollector(_BaseP2MetricsCollector):
    """P2 metrics with frame-level flight/candidate diagnostics for GT calibration."""

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
