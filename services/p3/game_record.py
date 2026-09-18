from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from core.p3_contract import P3SourceEvent


@dataclass(slots=True)
class GameRecordSnapshot:
    """Threshold-free structural summary of one P3 source-event set.

    This snapshot deliberately reports candidates and confirmed events separately. It is
    safe to build before the P2 accuracy exit gate because it describes what the source
    run emitted, not what CourtFlux should ultimately present as basketball truth.
    """

    game_id: str
    analysis_run_id: str
    event_count: int
    event_counts: dict[str, int] = field(default_factory=dict)
    decision_counts: dict[str, int] = field(default_factory=dict)
    evidence_event_count: int = 0
    attributed_event_count: int = 0
    timestamped_event_count: int = 0
    first_start_ms: int | None = None
    last_end_ms: int | None = None

    @property
    def evidence_coverage(self) -> float:
        return self.evidence_event_count / self.event_count if self.event_count else 0.0

    @property
    def attribution_coverage(self) -> float:
        return self.attributed_event_count / self.event_count if self.event_count else 0.0

    @property
    def timestamp_coverage(self) -> float:
        return self.timestamped_event_count / self.event_count if self.event_count else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "event_count": self.event_count,
            "event_counts": dict(self.event_counts),
            "decision_counts": dict(self.decision_counts),
            "evidence_event_count": self.evidence_event_count,
            "attributed_event_count": self.attributed_event_count,
            "timestamped_event_count": self.timestamped_event_count,
            "evidence_coverage": self.evidence_coverage,
            "attribution_coverage": self.attribution_coverage,
            "timestamp_coverage": self.timestamp_coverage,
            "first_start_ms": self.first_start_ms,
            "last_end_ms": self.last_end_ms,
        }


def build_game_record_snapshot(events: Iterable[P3SourceEvent]) -> GameRecordSnapshot:
    rows = list(events)
    if not rows:
        raise ValueError("build_game_record_snapshot requires at least one source event")

    game_ids = {row.game_id for row in rows}
    run_ids = {row.analysis_run_id for row in rows}
    if len(game_ids) != 1 or len(run_ids) != 1:
        raise ValueError("P3 Game Record cannot mix games or analysis runs")

    event_counts = Counter(row.event_type for row in rows)
    decision_counts = Counter(row.decision for row in rows)
    starts = [row.start_ms for row in rows if row.start_ms is not None]
    ends = [row.end_ms for row in rows if row.end_ms is not None]

    return GameRecordSnapshot(
        game_id=rows[0].game_id,
        analysis_run_id=rows[0].analysis_run_id,
        event_count=len(rows),
        event_counts=dict(sorted(event_counts.items())),
        decision_counts=dict(sorted(decision_counts.items())),
        evidence_event_count=sum(bool(row.evidence) for row in rows),
        attributed_event_count=sum(row.actor_game_player_id is not None for row in rows),
        timestamped_event_count=sum(
            row.start_ms is not None and row.end_ms is not None for row in rows
        ),
        first_start_ms=min(starts) if starts else None,
        last_end_ms=max(ends) if ends else None,
    )
