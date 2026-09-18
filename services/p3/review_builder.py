from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable

from core.p3_clip import ClipContextPolicy, P3ClipRequest, plan_evidence_clip
from core.p3_contract import P3SourceEvent


REVIEW_BUILDER_VERSION = "p3.review_builder.v1"
DEFAULT_PRESENTATION_ORDER = (
    "turnover",
    "shot",
    "dunk",
    "loose_ball",
    "rebound",
    "pass",
    "alley_oop",
)


@dataclass(slots=True)
class ReviewBuildPolicy:
    """Versioned policy for materializing review items from immutable source events.

    The default deliberately applies no confidence threshold. Product thresholds may be
    added through configuration later and therefore become part of the derivation
    fingerprint instead of being hidden in code.
    """

    version: str = REVIEW_BUILDER_VERSION
    include_support_states: tuple[str, ...] = ("candidate", "confirmed", "mixed")
    minimum_confidence_by_type: dict[str, float] = field(default_factory=dict)
    presentation_order: tuple[str, ...] = DEFAULT_PRESENTATION_ORDER
    clip_context_policy: ClipContextPolicy = field(default_factory=ClipContextPolicy)

    def minimum_confidence(self, review_type: str) -> float | None:
        raw = self.minimum_confidence_by_type.get(str(review_type))
        if raw is None:
            return None
        return max(0.0, min(float(raw), 1.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "include_support_states": list(self.include_support_states),
            "minimum_confidence_by_type": {
                key: max(0.0, min(float(value), 1.0))
                for key, value in sorted(self.minimum_confidence_by_type.items())
            },
            "presentation_order": list(self.presentation_order),
            "clip_context_policy": self.clip_context_policy.as_dict(),
        }


@dataclass(slots=True)
class ReviewItemDraft:
    game_id: str
    analysis_run_id: str
    review_type: str
    source_event_ids: tuple[str, ...]
    source_possession_id: str | None
    game_player_ids: tuple[str, ...]
    team_id: str | None
    confidence: float | None
    support_state: str
    completeness: str
    reason_shown: dict[str, Any]

    def identity_payload(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "review_type": self.review_type,
            "source_event_ids": list(self.source_event_ids),
            "source_possession_id": self.source_possession_id,
        }

    def review_key(self) -> str:
        payload = json.dumps(
            self.identity_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "review_key": self.review_key(),
            "game_player_ids": list(self.game_player_ids),
            "team_id": self.team_id,
            "confidence": self.confidence,
            "support_state": self.support_state,
            "completeness": self.completeness,
            "reason_shown": dict(self.reason_shown),
        }


@dataclass(slots=True)
class ReviewBuildResult:
    item: ReviewItemDraft
    clip_request: P3ClipRequest | None
    sort_key: tuple[int, int, str]


@dataclass(slots=True)
class ThingsToReviewSnapshot:
    game_id: str
    analysis_run_id: str
    policy: ReviewBuildPolicy
    items: tuple[ReviewBuildResult, ...]
    excluded_by_policy: dict[str, int] = field(default_factory=dict)

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def clip_plannable_count(self) -> int:
        return sum(1 for result in self.items if result.clip_request is not None)

    def counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.items:
            counts[result.item.review_type] = counts.get(result.item.review_type, 0) + 1
        return counts

    def counts_by_support_state(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.items:
            state = result.item.support_state
            counts[state] = counts.get(state, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "policy": self.policy.as_dict(),
            "total_items": self.total_items,
            "clip_plannable_count": self.clip_plannable_count,
            "counts_by_type": self.counts_by_type(),
            "counts_by_support_state": self.counts_by_support_state(),
            "excluded_by_policy": dict(self.excluded_by_policy),
            "items": [
                {
                    **result.item.as_dict(),
                    "clip_request": (
                        result.clip_request.as_dict() if result.clip_request is not None else None
                    ),
                }
                for result in self.items
            ],
        }


def _is_loose(event: P3SourceEvent) -> bool:
    return event.event_type in {
        "loose_ball_started",
        "loose_ball_recovered",
        "loose_ball_ended",
    }


def _review_type(event: P3SourceEvent) -> str | None:
    if event.event_type == "pass_candidate":
        return "pass"
    if event.event_type == "turnover_candidate":
        return "turnover"
    if event.event_type == "shot_attempt":
        shot_type = str(event.attributes.get("shot_type") or "").strip().lower()
        return "dunk" if shot_type == "dunk" else "shot"
    if event.event_type == "rebound_candidate":
        return "rebound"
    if _is_loose(event):
        return "loose_ball"
    return None


def _loose_groups(events: list[P3SourceEvent]) -> list[list[P3SourceEvent]]:
    starts = [event for event in events if event.event_type == "loose_ball_started"]
    terminals = [
        event
        for event in events
        if event.event_type in {"loose_ball_recovered", "loose_ball_ended"}
    ]
    terminal_groups: dict[int, list[P3SourceEvent]] = {}
    for terminal in terminals:
        terminal_groups.setdefault(terminal.frame_start, []).append(terminal)

    used_start_ids: set[str] = set()
    groups: list[list[P3SourceEvent]] = []
    for start_frame in sorted(terminal_groups):
        matching_starts = [
            event
            for event in starts
            if event.source_event_id not in used_start_ids and event.frame_end == start_frame
        ]
        for event in matching_starts:
            used_start_ids.add(event.source_event_id)
        group = matching_starts + sorted(
            terminal_groups[start_frame], key=lambda event: (event.frame_end, event.source_event_id)
        )
        groups.append(group)

    for event in starts:
        if event.source_event_id not in used_start_ids:
            groups.append([event])

    return groups


def _support_state(events: list[P3SourceEvent]) -> str:
    decisions = {event.decision for event in events}
    if decisions == {"confirmed"}:
        return "confirmed"
    if decisions == {"candidate"}:
        return "candidate"
    return "mixed"


def _completeness(events: list[P3SourceEvent]) -> str:
    if any(event.event_type == "loose_ball_started" for event in events) and not any(
        event.event_type in {"loose_ball_recovered", "loose_ball_ended"}
        for event in events
    ):
        return "fragmented"
    if all(event.start_ms is not None and event.end_ms is not None for event in events):
        return "complete"
    return "unknown"


def _source_possession_id(events: list[P3SourceEvent]) -> str | None:
    values = {
        str(value)
        for event in events
        for value in (
            event.attributes.get("possession_id"),
            event.attributes.get("source_possession_id"),
        )
        if value is not None and str(value)
    }
    return next(iter(values)) if len(values) == 1 else None


def _team_id(events: list[P3SourceEvent]) -> str | None:
    teams = {str(event.team_id) for event in events if event.team_id is not None}
    return next(iter(teams)) if len(teams) == 1 else None


def _game_player_ids(events: list[P3SourceEvent]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                player_id
                for event in events
                for player_id in (
                    event.actor_game_player_id,
                    event.secondary_game_player_id,
                )
                if player_id is not None
            }
        )
    )


def _sort_key(events: list[P3SourceEvent], review_type: str) -> tuple[int, int, str]:
    starts = [int(event.start_ms) for event in events if event.start_ms is not None]
    if starts:
        return (0, min(starts), review_type)
    return (1, min(event.frame_start for event in events), review_type)


def _draft_for_group(
    events: list[P3SourceEvent],
    *,
    review_type: str,
    policy: ReviewBuildPolicy,
) -> ReviewItemDraft:
    confidence = min((event.confidence for event in events), default=None)
    support_state = _support_state(events)
    evidence = sorted({item for event in events for item in event.evidence})
    reason: dict[str, Any] = {
        "builder_version": policy.version,
        "source_event_types": [event.event_type for event in events],
        "source_decisions": [event.decision for event in events],
        "source_confidences": [event.confidence for event in events],
        "confidence_aggregation": "minimum_source_confidence",
        "evidence": evidence,
    }
    threshold = policy.minimum_confidence(review_type)
    if threshold is not None:
        reason["configured_minimum_confidence"] = threshold

    return ReviewItemDraft(
        game_id=events[0].game_id,
        analysis_run_id=events[0].analysis_run_id,
        review_type=review_type,
        source_event_ids=tuple(event.source_event_id for event in events),
        source_possession_id=_source_possession_id(events),
        game_player_ids=_game_player_ids(events),
        team_id=_team_id(events),
        confidence=confidence,
        support_state=support_state,
        completeness=_completeness(events),
        reason_shown=reason,
    )


def build_things_to_review(
    events: Iterable[P3SourceEvent],
    *,
    policy: ReviewBuildPolicy | None = None,
    source_duration_ms: int | None = None,
    game_id: str | None = None,
    analysis_run_id: str | None = None,
) -> ThingsToReviewSnapshot:
    rows = list(events)
    policy = policy or ReviewBuildPolicy()

    if rows:
        game_ids = {event.game_id for event in rows}
        run_ids = {event.analysis_run_id for event in rows}
        if len(game_ids) != 1 or len(run_ids) != 1:
            raise ValueError("Things to Review cannot mix games or analysis runs")
        resolved_game_id = next(iter(game_ids))
        resolved_run_id = next(iter(run_ids))
        if game_id is not None and str(game_id) != resolved_game_id:
            raise ValueError("provided game_id does not match source events")
        if analysis_run_id is not None and str(analysis_run_id) != resolved_run_id:
            raise ValueError("provided analysis_run_id does not match source events")
    else:
        if game_id is None or analysis_run_id is None:
            raise ValueError("empty Things to Review input requires game_id and analysis_run_id")
        resolved_game_id = str(game_id)
        resolved_run_id = str(analysis_run_id)

    groups: list[list[P3SourceEvent]] = []
    groups.extend([[event] for event in rows if not _is_loose(event) and _review_type(event)])
    groups.extend(_loose_groups([event for event in rows if _is_loose(event)]))

    results: list[ReviewBuildResult] = []
    excluded: dict[str, int] = {}

    for group in groups:
        review_type = _review_type(group[-1])
        if review_type is None:
            continue
        draft = _draft_for_group(group, review_type=review_type, policy=policy)

        if draft.support_state not in policy.include_support_states:
            key = f"support_state:{draft.support_state}"
            excluded[key] = excluded.get(key, 0) + 1
            continue

        minimum_confidence = policy.minimum_confidence(review_type)
        if (
            minimum_confidence is not None
            and draft.confidence is not None
            and draft.confidence < minimum_confidence
        ):
            key = f"minimum_confidence:{review_type}"
            excluded[key] = excluded.get(key, 0) + 1
            continue

        clip_request: P3ClipRequest | None = None
        try:
            clip_request = plan_evidence_clip(
                group,
                review_type=review_type,
                policy=policy.clip_context_policy,
                source_duration_ms=source_duration_ms,
            )
        except ValueError:
            clip_request = None

        draft.reason_shown["clip_plannable"] = clip_request is not None
        results.append(
            ReviewBuildResult(
                item=draft,
                clip_request=clip_request,
                sort_key=_sort_key(group, review_type),
            )
        )

    results.sort(key=lambda result: result.sort_key)
    return ThingsToReviewSnapshot(
        game_id=resolved_game_id,
        analysis_run_id=resolved_run_id,
        policy=policy,
        items=tuple(results),
        excluded_by_policy=excluded,
    )
