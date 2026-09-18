from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from core.p3_contract import P3SourceEvent


DEFAULT_CLIP_CONTEXT_MS: dict[str, tuple[int, int]] = {
    "pass": (4000, 4000),
    "shot": (6000, 4000),
    "turnover": (7000, 5000),
    "rebound": (5000, 4000),
    "loose_ball": (6000, 5000),
    "dunk": (7000, 5000),
    "alley_oop": (8000, 5000),
}


@dataclass(slots=True)
class ClipContextPolicy:
    """Versioned review-context configuration, independent from event truth."""

    version: str = "p3.clip_context.v1"
    windows_ms: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(DEFAULT_CLIP_CONTEXT_MS)
    )
    fallback_ms: tuple[int, int] = (5000, 4000)

    def context_for(self, review_type: str) -> tuple[int, int]:
        pre_ms, post_ms = self.windows_ms.get(str(review_type), self.fallback_ms)
        return max(0, int(pre_ms)), max(0, int(post_ms))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "windows_ms": {
                key: [int(value[0]), int(value[1])]
                for key, value in sorted(self.windows_ms.items())
            },
            "fallback_ms": [int(self.fallback_ms[0]), int(self.fallback_ms[1])],
        }


@dataclass(slots=True)
class P3ClipRequest:
    """Regenerable physical-clip request anchored to source-video time."""

    game_id: str
    analysis_run_id: str
    review_type: str
    source_start_ms: int
    source_end_ms: int
    clip_start_ms: int
    clip_end_ms: int
    source_event_ids: tuple[str, ...]
    game_player_ids: tuple[str, ...] = ()
    context_policy_version: str = "p3.clip_context.v1"
    generation_version: str = "p3.clip_generation.v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "review_type": self.review_type,
            "source_start_ms": self.source_start_ms,
            "source_end_ms": self.source_end_ms,
            "clip_start_ms": self.clip_start_ms,
            "clip_end_ms": self.clip_end_ms,
            "source_event_ids": list(self.source_event_ids),
            "game_player_ids": list(self.game_player_ids),
            "context_policy_version": self.context_policy_version,
            "generation_version": self.generation_version,
        }


def plan_evidence_clip(
    events: Iterable[P3SourceEvent],
    *,
    review_type: str,
    policy: ClipContextPolicy | None = None,
    source_duration_ms: int | None = None,
    generation_version: str = "p3.clip_generation.v1",
) -> P3ClipRequest:
    """Plan a physical evidence clip without changing the underlying event decision."""

    rows = list(events)
    if not rows:
        raise ValueError("plan_evidence_clip requires at least one source event")

    game_ids = {row.game_id for row in rows}
    run_ids = {row.analysis_run_id for row in rows}
    if len(game_ids) != 1 or len(run_ids) != 1:
        raise ValueError("evidence clip cannot mix games or analysis runs")

    timestamped = [
        row for row in rows if row.start_ms is not None and row.end_ms is not None
    ]
    if not timestamped:
        raise ValueError("evidence clip requires source-video timestamps")

    source_start_ms = min(int(row.start_ms) for row in timestamped if row.start_ms is not None)
    source_end_ms = max(int(row.end_ms) for row in timestamped if row.end_ms is not None)
    policy = policy or ClipContextPolicy()
    pre_ms, post_ms = policy.context_for(review_type)

    clip_start_ms = max(0, source_start_ms - pre_ms)
    clip_end_ms = source_end_ms + post_ms
    if source_duration_ms is not None:
        clip_end_ms = min(max(0, int(source_duration_ms)), clip_end_ms)
    clip_end_ms = max(clip_start_ms, clip_end_ms)

    game_players = sorted(
        {
            player_id
            for row in rows
            for player_id in (
                row.actor_game_player_id,
                row.secondary_game_player_id,
            )
            if player_id is not None
        }
    )

    return P3ClipRequest(
        game_id=rows[0].game_id,
        analysis_run_id=rows[0].analysis_run_id,
        review_type=str(review_type),
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
        clip_start_ms=clip_start_ms,
        clip_end_ms=clip_end_ms,
        source_event_ids=tuple(row.source_event_id for row in rows),
        game_player_ids=tuple(game_players),
        context_policy_version=policy.version,
        generation_version=str(generation_version),
    )
