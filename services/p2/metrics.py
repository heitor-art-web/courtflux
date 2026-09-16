from __future__ import annotations

from collections import Counter
from typing import Any


class P2MetricsCollector:
    """Aggregate P2 runtime metrics without changing basketball decisions."""

    UNATTRIBUTED_TEAM = "unattributed"

    def __init__(self) -> None:
        self.frames_seen = 0
        self.p2_processed_frames = 0
        self.metric_eligible_frames = 0
        self.engine_eligible_frames = 0
        self.possessed_frames = 0
        self.handler_memory_frames = 0
        self.possession_confidence_total = 0.0
        self.possession_confidence_samples = 0
        self.p1_not_mutated_frames = 0
        self.event_counts: Counter[str] = Counter()
        self.decision_counts: Counter[str] = Counter()
        self.shot_outcomes: Counter[str] = Counter()
        self.event_confidence_total: Counter[str] = Counter()
        self.event_confidence_samples: Counter[str] = Counter()
        self.control_state_counts: Counter[str] = Counter()
        self.reset_reasons: Counter[str] = Counter()
        self.loose_terminal_reasons: Counter[str] = Counter()
        self.team_stats: dict[str, Counter[str]] = {}
        self.events_with_actor = 0
        self.events_with_team = 0
        self.event_actor_counts: Counter[str] = Counter()
        self.event_team_counts: Counter[str] = Counter()
        self.total_events = 0
        self.fps = 30.0
        self.max_reset_count = 0
        self.event_rows: list[dict[str, Any]] = []
        self.state_rows: list[dict[str, Any]] = []

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(100.0 * numerator / max(denominator, 1), 2)

    @classmethod
    def _team_key(cls, raw_team_id: Any) -> str:
        if raw_team_id is None:
            return cls.UNATTRIBUTED_TEAM
        value = str(raw_team_id).strip()
        if not value or value.lower() in {"unknown", "none", "null"}:
            return cls.UNATTRIBUTED_TEAM
        return value

    def _team_counter(self, raw_team_id: Any) -> Counter[str]:
        team_key = self._team_key(raw_team_id)
        if team_key not in self.team_stats:
            self.team_stats[team_key] = Counter()
        return self.team_stats[team_key]

    def observe(self, frame_state: Any) -> None:
        self.frames_seen += 1
        try:
            self.fps = max(1.0, float(getattr(frame_state, "fps", self.fps) or self.fps))
        except (TypeError, ValueError):
            pass

        # Metric eligibility stays conservative (wide/court-visible analysis scope).
        # P2 engine eligibility is tracked separately and may include GAMEPLAY_LIMITED.
        metric_eligible = bool(getattr(frame_state, "valid_gameplay", False))
        if metric_eligible:
            self.metric_eligible_frames += 1

        court_state = getattr(frame_state, "court_state", None)
        gameplay_present = (
            bool(getattr(court_state, "gameplay_present"))
            if court_state is not None and getattr(court_state, "gameplay_present", None) is not None
            else metric_eligible
        )
        gameplay_status = (
            str(getattr(court_state, "gameplay_status", "unknown"))
            if court_state is not None
            else ("gameplay_analysis" if metric_eligible else "unknown")
        )

        diagnostics = dict(getattr(frame_state, "diagnostics", {}) or {})
        p2 = dict(diagnostics.get("p2", {}) or {})
        if not p2:
            return
        self.p2_processed_frames += 1

        if bool(p2.get("p1_state_is_not_mutated", False)):
            self.p1_not_mutated_frames += 1

        engine_eligible = bool(p2.get("engine_eligible", gameplay_present))
        if engine_eligible:
            self.engine_eligible_frames += 1

        possession = dict(p2.get("possession", {}) or {})
        current_track_id = possession.get("current_track_id")
        control_state = str(possession.get("control_state") or "unknown")
        if engine_eligible:
            self.control_state_counts[control_state] += 1

        # Possession/control metrics intentionally use P2 engine scope rather than the
        # stricter spatial-analysis scope. This keeps GAMEPLAY_LIMITED continuity visible.
        if engine_eligible and current_track_id is not None:
            self.handler_memory_frames += 1

        if engine_eligible and control_state == "possessed":
            self.possessed_frames += 1
            self._team_counter(possession.get("current_team_id"))["possessed_frames"] += 1
            try:
                confidence = float(possession.get("current_confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            self.possession_confidence_total += confidence
            self.possession_confidence_samples += 1

        reset = dict(p2.get("temporal_reset", {}) or {})
        try:
            reset_count = max(0, int(reset.get("count", 0) or 0))
        except (TypeError, ValueError):
            reset_count = 0
        if reset_count > self.max_reset_count:
            delta = reset_count - self.max_reset_count
            reason = str(reset.get("last_reason") or "unknown")
            self.reset_reasons[reason] += delta
            self.max_reset_count = reset_count

        shot_diag = dict(p2.get("shot", {}) or {})
        flight = dict(possession.get("flight", {}) or {})
        self.state_rows.append(
            {
                "frame_id": int(getattr(frame_state, "frame_id", 0)),
                "timestamp_sec": float(getattr(frame_state, "timestamp", 0.0) or 0.0),
                "metric_eligible": metric_eligible,
                "gameplay_present": gameplay_present,
                "gameplay_status": gameplay_status,
                "engine_eligible": engine_eligible,
                "control_state": control_state,
                "current_track_id": current_track_id,
                "current_team_id": possession.get("current_team_id"),
                "current_confidence": possession.get("current_confidence"),
                "lost_frames": possession.get("lost_frames"),
                "flight_active": bool(flight.get("active", False)),
                "flight_supported": bool(flight.get("flight_supported", False)),
                "flight_frames": flight.get("flight_frames"),
                "shot_active": bool(shot_diag.get("active", False)),
                "shot_awaiting_rearm": bool(shot_diag.get("awaiting_rearm", False)),
                "shot_rearm_clear_observed_frames": shot_diag.get("rearm_clear_observed_frames"),
                "reset_count": reset_count,
                "reset_reason": reset.get("last_reason"),
            }
        )

        for event in list(p2.get("events", []) or []):
            if not isinstance(event, dict):
                continue
            row = dict(event)
            event_type = str(row.get("event_type") or "unknown")
            decision = str(row.get("decision") or "unknown")
            self.event_counts[event_type] += 1
            self.decision_counts[decision] += 1
            self.total_events += 1

            try:
                confidence = float(row.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            self.event_confidence_total[event_type] += confidence
            self.event_confidence_samples[event_type] += 1

            if row.get("actor_track_id") is not None:
                self.events_with_actor += 1
                self.event_actor_counts[event_type] += 1
            if row.get("team_id") not in {None, "", "unknown", "UNKNOWN"}:
                self.events_with_team += 1
                self.event_team_counts[event_type] += 1

            attributes = dict(row.get("attributes", {}) or {})
            team = self._team_counter(row.get("team_id"))
            team["events"] += 1

            if event_type == "possession_acquired":
                team["possession_acquisitions"] += 1
            elif event_type == "possession_change":
                team["possession_changes"] += 1
            elif event_type == "pass_candidate":
                team["passes"] += 1
                if bool(attributes.get("flight_supported", False)):
                    team["passes_with_flight_evidence"] += 1
            elif event_type == "turnover_candidate":
                team["turnovers"] += 1
                if bool(attributes.get("flight_supported", False)):
                    team["turnovers_with_flight_evidence"] += 1
                else:
                    team["direct_turnovers"] += 1
            elif event_type == "rebound_candidate":
                team["rebounds"] += 1
            elif event_type == "shot_attempt":
                outcome = str(attributes.get("outcome") or "unknown")
                self.shot_outcomes[outcome] += 1
                team["shot_attempts"] += 1
                team[f"shot_{outcome}"] += 1
            elif event_type == "loose_ball_ended":
                self.loose_terminal_reasons[
                    str(attributes.get("terminal_reason") or "unknown")
                ] += 1

            self.event_rows.append(row)

    def _shot_retrigger_metrics(self) -> dict[str, Any]:
        shots = [
            row
            for row in self.event_rows
            if str(row.get("event_type") or "") == "shot_attempt"
        ]
        shots.sort(key=lambda row: int(row.get("frame_start", 0)))
        threshold_frames = max(1, int(round(self.fps * 0.30)))
        rapid_unknown_retriggers = 0
        min_gap_frames: int | None = None
        for previous, current in zip(shots, shots[1:]):
            previous_outcome = str(
                dict(previous.get("attributes", {}) or {}).get("outcome") or "unknown"
            )
            gap = int(current.get("frame_start", 0)) - int(previous.get("frame_end", 0))
            if min_gap_frames is None or gap < min_gap_frames:
                min_gap_frames = gap
            if previous_outcome == "unknown" and 0 <= gap <= threshold_frames:
                rapid_unknown_retriggers += 1
        return {
            "rapid_unknown_retrigger_window_frames": threshold_frames,
            "rapid_unknown_retrigger_count": rapid_unknown_retriggers,
            "minimum_inter_shot_gap_frames": min_gap_frames,
        }

    def _team_summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for team_id in sorted(
            self.team_stats,
            key=lambda value: (value == self.UNATTRIBUTED_TEAM, value.lower()),
        ):
            stats = self.team_stats[team_id]
            result[team_id] = {
                "possessed_frames": int(stats.get("possessed_frames", 0)),
                "possession_share_rate": self._rate(
                    int(stats.get("possessed_frames", 0)),
                    self.possessed_frames,
                ),
                "events": int(stats.get("events", 0)),
                "possession_acquisitions": int(stats.get("possession_acquisitions", 0)),
                "possession_changes": int(stats.get("possession_changes", 0)),
                "passes": int(stats.get("passes", 0)),
                "passes_with_flight_evidence": int(
                    stats.get("passes_with_flight_evidence", 0)
                ),
                "turnovers": int(stats.get("turnovers", 0)),
                "turnovers_with_flight_evidence": int(
                    stats.get("turnovers_with_flight_evidence", 0)
                ),
                "direct_turnovers": int(stats.get("direct_turnovers", 0)),
                "rebounds": int(stats.get("rebounds", 0)),
                "shots": {
                    "attempts": int(stats.get("shot_attempts", 0)),
                    "outcomes": {
                        "made": int(stats.get("shot_made", 0)),
                        "missed": int(stats.get("shot_missed", 0)),
                        "unknown": int(stats.get("shot_unknown", 0)),
                    },
                },
            }
        return result

    def summary(self) -> dict[str, Any]:
        avg_possession_confidence = (
            self.possession_confidence_total / self.possession_confidence_samples
            if self.possession_confidence_samples
            else None
        )
        event_confidence_mean = {
            event_type: round(
                self.event_confidence_total[event_type]
                / max(self.event_confidence_samples[event_type], 1),
                4,
            )
            for event_type in sorted(self.event_confidence_samples)
        }
        attribution_by_type = {
            event_type: {
                "total": int(count),
                "actor_count": int(self.event_actor_counts.get(event_type, 0)),
                "actor_rate": self._rate(
                    int(self.event_actor_counts.get(event_type, 0)),
                    int(count),
                ),
                "team_count": int(self.event_team_counts.get(event_type, 0)),
                "team_rate": self._rate(
                    int(self.event_team_counts.get(event_type, 0)),
                    int(count),
                ),
            }
            for event_type, count in sorted(self.event_counts.items())
        }
        shot_attempts = int(self.event_counts.get("shot_attempt", 0))
        unknown_shots = int(self.shot_outcomes.get("unknown", 0))

        passes_with_flight = 0
        turnovers_with_flight = 0
        direct_turnovers = 0
        for row in self.event_rows:
            event_type = str(row.get("event_type") or "")
            attrs = dict(row.get("attributes", {}) or {})
            flight_supported = bool(attrs.get("flight_supported", False))
            if event_type == "pass_candidate" and flight_supported:
                passes_with_flight += 1
            if event_type == "turnover_candidate":
                if flight_supported:
                    turnovers_with_flight += 1
                else:
                    direct_turnovers += 1

        control_rates = {
            state: self._rate(count, self.engine_eligible_frames)
            for state, count in sorted(self.control_state_counts.items())
        }
        loose_started = int(self.event_counts.get("loose_ball_started", 0))
        loose_recovered = int(self.event_counts.get("loose_ball_recovered", 0))
        loose_ended = int(self.event_counts.get("loose_ball_ended", 0))
        loose_closed = loose_recovered + loose_ended
        loose_open = max(0, loose_started - loose_closed)

        return {
            "version": "p2_metrics_v5",
            "frames_seen": self.frames_seen,
            "p2_processed_frames": self.p2_processed_frames,
            "metric_eligible_frames": self.metric_eligible_frames,
            "engine_eligible_frames": self.engine_eligible_frames,
            "scope": {
                "metric_policy": "court_state_analysis_eligible",
                "engine_policy": "court_state_gameplay_present",
                "limited_gameplay_frames": max(
                    0,
                    self.engine_eligible_frames - self.metric_eligible_frames,
                ),
            },
            "control_state": {
                "counts": dict(sorted(self.control_state_counts.items())),
                "rates": control_rates,
            },
            "possession": {
                "possessed_frames": self.possessed_frames,
                "possessed_rate": self._rate(
                    self.possessed_frames,
                    self.engine_eligible_frames,
                ),
                "handler_memory_frames": self.handler_memory_frames,
                "handler_memory_rate": self._rate(
                    self.handler_memory_frames,
                    self.engine_eligible_frames,
                ),
                "avg_possession_confidence": (
                    None
                    if avg_possession_confidence is None
                    else round(avg_possession_confidence, 4)
                ),
                "changes": int(self.event_counts.get("possession_change", 0)),
                "acquisitions": int(self.event_counts.get("possession_acquired", 0)),
            },
            "events": {
                "total": self.total_events,
                "counts": dict(sorted(self.event_counts.items())),
                "decisions": dict(sorted(self.decision_counts.items())),
                "mean_confidence_by_type": event_confidence_mean,
                "actor_attribution_rate": self._rate(
                    self.events_with_actor,
                    self.total_events,
                ),
                "team_attribution_rate": self._rate(
                    self.events_with_team,
                    self.total_events,
                ),
                "attribution_by_event_type": attribution_by_type,
            },
            "teams": self._team_summary(),
            "shots": {
                "attempts": shot_attempts,
                "outcomes": dict(sorted(self.shot_outcomes.items())),
                "unknown_outcome_rate": self._rate(unknown_shots, shot_attempts),
                **self._shot_retrigger_metrics(),
            },
            "transitions": {
                "passes": int(self.event_counts.get("pass_candidate", 0)),
                "passes_with_flight_evidence": passes_with_flight,
                "turnovers": int(self.event_counts.get("turnover_candidate", 0)),
                "turnovers_with_flight_evidence": turnovers_with_flight,
                "direct_turnovers": direct_turnovers,
                "loose_ball_started": loose_started,
                "loose_ball_recovered": loose_recovered,
                "loose_ball_ended": loose_ended,
                "loose_ball_closed": loose_closed,
                "loose_ball_open_intervals": loose_open,
                "loose_ball_closure_rate": self._rate(loose_closed, loose_started),
                "loose_ball_terminal_reasons": dict(sorted(self.loose_terminal_reasons.items())),
            },
            "passes": int(self.event_counts.get("pass_candidate", 0)),
            "turnovers": int(self.event_counts.get("turnover_candidate", 0)),
            "rebounds": int(self.event_counts.get("rebound_candidate", 0)),
            "resets": {
                "count": self.max_reset_count,
                "reasons": dict(sorted(self.reset_reasons.items())),
            },
            "p1_state_not_mutated_rate": self._rate(
                self.p1_not_mutated_frames,
                self.p2_processed_frames,
            ),
        }
