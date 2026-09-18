from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable

from core.p3_contract import P3SourceEvent


P3_CLIP_REQUEST_CONTRACT_VERSION = "p3.clip_request.v1"
DEFAULT_CLIP_CONTEXT_MS: dict[str, tuple[int, int]] = {
    "pass": (4000, 4000),
    "shot": (6000, 4000),
    "turnover": (7000, 5000),
    "rebound": (5000, 4000),
    "loose_ball": (6000, 5000),
    "dunk": (7000, 5000),
    "alley_oop": (8000, 5000),
}


def _required_text(value: Any, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _optional_text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, name)


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def analysis_run_source_ref(analysis_run_id: str) -> str:
    """Stable fallback source identity resolved through ``analysis_runs.video_source``."""

    return f"analysis-run:{_required_text(analysis_run_id, 'analysis_run_id')}"


@dataclass(slots=True)
class ClipContextPolicy:
    """Versioned review-context configuration, independent from basketball truth."""

    version: str = "p3.clip_context.v1"
    windows_ms: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(DEFAULT_CLIP_CONTEXT_MS)
    )
    fallback_ms: tuple[int, int] = (5000, 4000)

    def __post_init__(self) -> None:
        self.version = _required_text(self.version, "clip context policy version")
        normalized: dict[str, tuple[int, int]] = {}
        for review_type, raw_window in self.windows_ms.items():
            key = _required_text(review_type, "review_type")
            if not isinstance(raw_window, (list, tuple)) or len(raw_window) != 2:
                raise ValueError(f"clip context for {key!r} must contain pre/post milliseconds")
            normalized[key] = (
                _non_negative_int(raw_window[0], f"{key}.pre_ms"),
                _non_negative_int(raw_window[1], f"{key}.post_ms"),
            )
        self.windows_ms = normalized
        if not isinstance(self.fallback_ms, (list, tuple)) or len(self.fallback_ms) != 2:
            raise ValueError("fallback_ms must contain pre/post milliseconds")
        self.fallback_ms = (
            _non_negative_int(self.fallback_ms[0], "fallback.pre_ms"),
            _non_negative_int(self.fallback_ms[1], "fallback.post_ms"),
        )

    def context_for(self, review_type: str) -> tuple[int, int]:
        return self.windows_ms.get(_required_text(review_type, "review_type"), self.fallback_ms)

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
    """Regenerable physical-clip request anchored to a stable source-video reference."""

    game_id: str
    analysis_run_id: str
    source_video_ref: str
    review_type: str
    source_start_ms: int
    source_end_ms: int
    clip_start_ms: int
    clip_end_ms: int
    source_event_ids: tuple[str, ...]
    source_possession_id: str | None = None
    game_player_ids: tuple[str, ...] = ()
    context_policy_version: str = "p3.clip_context.v1"
    generation_version: str = "p3.clip_generation.v1"
    contract_version: str = P3_CLIP_REQUEST_CONTRACT_VERSION

    def __post_init__(self) -> None:
        self.game_id = _required_text(self.game_id, "game_id")
        self.analysis_run_id = _required_text(self.analysis_run_id, "analysis_run_id")
        self.source_video_ref = _required_text(self.source_video_ref, "source_video_ref")
        self.review_type = _required_text(self.review_type, "review_type")
        self.source_start_ms = _non_negative_int(self.source_start_ms, "source_start_ms")
        self.source_end_ms = _non_negative_int(self.source_end_ms, "source_end_ms")
        self.clip_start_ms = _non_negative_int(self.clip_start_ms, "clip_start_ms")
        self.clip_end_ms = _non_negative_int(self.clip_end_ms, "clip_end_ms")
        if self.source_end_ms < self.source_start_ms:
            raise ValueError("source_end_ms must be >= source_start_ms")
        if self.clip_end_ms <= self.clip_start_ms:
            raise ValueError("clip_end_ms must be > clip_start_ms")
        if self.clip_start_ms > self.source_start_ms or self.clip_end_ms < self.source_end_ms:
            raise ValueError("physical clip bounds must contain the full source evidence interval")

        if not isinstance(self.source_event_ids, (list, tuple)) or not self.source_event_ids:
            raise ValueError("source_event_ids must contain at least one event id")
        event_ids = tuple(_required_text(value, "source_event_id") for value in self.source_event_ids)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("source_event_ids must be unique within one clip request")
        self.source_event_ids = event_ids

        self.source_possession_id = _optional_text(
            self.source_possession_id, "source_possession_id"
        )
        if not isinstance(self.game_player_ids, (list, tuple)):
            raise ValueError("game_player_ids must be an array")
        player_ids = tuple(_required_text(value, "game_player_id") for value in self.game_player_ids)
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("game_player_ids must be unique within one clip request")
        self.game_player_ids = player_ids
        self.context_policy_version = _required_text(
            self.context_policy_version, "context_policy_version"
        )
        self.generation_version = _required_text(
            self.generation_version, "generation_version"
        )
        self.contract_version = _required_text(self.contract_version, "contract_version")
        if self.contract_version != P3_CLIP_REQUEST_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported clip request contract {self.contract_version!r}; "
                f"expected {P3_CLIP_REQUEST_CONTRACT_VERSION!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "source_video_ref": self.source_video_ref,
            "review_type": self.review_type,
            "source_start_ms": self.source_start_ms,
            "source_end_ms": self.source_end_ms,
            "clip_start_ms": self.clip_start_ms,
            "clip_end_ms": self.clip_end_ms,
            "source_event_ids": list(self.source_event_ids),
            "source_possession_id": self.source_possession_id,
            "game_player_ids": list(self.game_player_ids),
            "context_policy_version": self.context_policy_version,
            "generation_version": self.generation_version,
            "contract_version": self.contract_version,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def plan_evidence_clip(
    events: Iterable[P3SourceEvent],
    *,
    review_type: str,
    policy: ClipContextPolicy | None = None,
    source_duration_ms: int | None = None,
    source_video_ref: str | None = None,
    source_possession_id: str | None = None,
    generation_version: str = "p3.clip_generation.v1",
) -> P3ClipRequest:
    """Plan one evidence clip without changing or repairing source-event truth."""

    rows = list(events)
    if not rows:
        raise ValueError("plan_evidence_clip requires at least one source event")

    game_ids = {row.game_id for row in rows}
    run_ids = {row.analysis_run_id for row in rows}
    if len(game_ids) != 1 or len(run_ids) != 1:
        raise ValueError("evidence clip cannot mix games or analysis runs")

    if any(row.start_ms is None or row.end_ms is None for row in rows):
        raise ValueError("evidence clip requires source-video timestamps for every source event")

    source_start_ms = min(int(row.start_ms) for row in rows if row.start_ms is not None)
    source_end_ms = max(int(row.end_ms) for row in rows if row.end_ms is not None)
    policy = policy or ClipContextPolicy()
    pre_ms, post_ms = policy.context_for(review_type)

    duration_ms: int | None = None
    if source_duration_ms is not None:
        duration_ms = _non_negative_int(source_duration_ms, "source_duration_ms")
        if duration_ms <= 0:
            raise ValueError("source_duration_ms must be positive when provided")
        if source_start_ms >= duration_ms or source_end_ms > duration_ms:
            raise ValueError("source evidence interval falls outside the source-video duration")

    clip_start_ms = max(0, source_start_ms - pre_ms)
    clip_end_ms = source_end_ms + post_ms
    if duration_ms is not None:
        clip_end_ms = min(duration_ms, clip_end_ms)

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
    run_id = rows[0].analysis_run_id

    return P3ClipRequest(
        game_id=rows[0].game_id,
        analysis_run_id=run_id,
        source_video_ref=(
            _required_text(source_video_ref, "source_video_ref")
            if source_video_ref is not None
            else analysis_run_source_ref(run_id)
        ),
        review_type=str(review_type),
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
        clip_start_ms=clip_start_ms,
        clip_end_ms=clip_end_ms,
        source_event_ids=tuple(row.source_event_id for row in rows),
        source_possession_id=source_possession_id,
        game_player_ids=tuple(game_players),
        context_policy_version=policy.version,
        generation_version=str(generation_version),
    )
