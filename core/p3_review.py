from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HumanReviewState(str, Enum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class ReviewCorrectionType(str, Enum):
    PASS = "pass"
    TURNOVER = "turnover"
    SHOT = "shot"
    REBOUND = "rebound"
    LOOSE_BALL = "loose_ball"
    DUNK = "dunk"
    ALLEY_OOP = "alley_oop"
    NONE = "none"


@dataclass(slots=True)
class P3HumanReview:
    """Human feedback stored separately from immutable model/source predictions."""

    review_item_id: str
    reviewer_id: str
    state: HumanReviewState
    corrected_event_type: ReviewCorrectionType | None = None
    corrected_game_player_id: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        self.review_item_id = str(self.review_item_id)
        self.reviewer_id = str(self.reviewer_id)
        if not isinstance(self.state, HumanReviewState):
            self.state = HumanReviewState(str(self.state))
        if self.corrected_event_type is not None and not isinstance(
            self.corrected_event_type, ReviewCorrectionType
        ):
            self.corrected_event_type = ReviewCorrectionType(
                str(self.corrected_event_type)
            )
        if self.corrected_game_player_id is not None:
            self.corrected_game_player_id = str(self.corrected_game_player_id)
        if self.note is not None:
            self.note = str(self.note)

        if self.state is HumanReviewState.CONFIRMED:
            if self.corrected_event_type is not None or self.corrected_game_player_id is not None:
                raise ValueError("confirmed review cannot carry correction fields")
        elif self.state is HumanReviewState.CORRECTED:
            if self.corrected_event_type is ReviewCorrectionType.NONE:
                raise ValueError("NONE is an explicit rejection, not a corrected event")
            if self.corrected_event_type is None and self.corrected_game_player_id is None:
                raise ValueError("corrected review requires an event or player correction")
        elif self.state is HumanReviewState.REJECTED:
            if self.corrected_event_type is not ReviewCorrectionType.NONE:
                raise ValueError("rejected review must use corrected_event_type=NONE")
            if self.corrected_game_player_id is not None:
                raise ValueError("rejected review cannot assign a corrected player")

    def as_dict(self) -> dict[str, Any]:
        return {
            "review_item_id": self.review_item_id,
            "reviewer_id": self.reviewer_id,
            "review_state": self.state.value,
            "corrected_event_type": (
                self.corrected_event_type.value
                if self.corrected_event_type is not None
                else None
            ),
            "corrected_game_player_id": self.corrected_game_player_id,
            "note": self.note,
        }
