from __future__ import annotations

from core.p3_review import HumanReviewState, P3HumanReview, ReviewCorrectionType


def test_confirmed_review_has_no_correction_fields() -> None:
    review = P3HumanReview(
        review_item_id="review-1",
        reviewer_id="user-1",
        state=HumanReviewState.CONFIRMED,
    )
    assert review.as_dict()["review_state"] == "confirmed"
    assert review.corrected_event_type is None
    assert review.corrected_game_player_id is None


def test_corrected_review_can_change_event_or_player() -> None:
    event_fix = P3HumanReview(
        review_item_id="review-1",
        reviewer_id="user-1",
        state=HumanReviewState.CORRECTED,
        corrected_event_type=ReviewCorrectionType.PASS,
    )
    player_fix = P3HumanReview(
        review_item_id="review-2",
        reviewer_id="user-1",
        state=HumanReviewState.CORRECTED,
        corrected_game_player_id="game_player_05",
    )
    assert event_fix.as_dict()["corrected_event_type"] == "pass"
    assert player_fix.as_dict()["corrected_game_player_id"] == "game_player_05"


def test_rejected_review_is_explicit_none_not_unreviewed() -> None:
    review = P3HumanReview(
        review_item_id="review-1",
        reviewer_id="user-1",
        state=HumanReviewState.REJECTED,
        corrected_event_type=ReviewCorrectionType.NONE,
    )
    assert review.as_dict()["review_state"] == "rejected"
    assert review.as_dict()["corrected_event_type"] == "none"


def test_invalid_review_states_fail_closed() -> None:
    invalid_cases = [
        {
            "state": HumanReviewState.CONFIRMED,
            "corrected_event_type": ReviewCorrectionType.SHOT,
        },
        {
            "state": HumanReviewState.CORRECTED,
        },
        {
            "state": HumanReviewState.CORRECTED,
            "corrected_event_type": ReviewCorrectionType.NONE,
        },
        {
            "state": HumanReviewState.REJECTED,
            "corrected_event_type": ReviewCorrectionType.TURNOVER,
        },
    ]
    for index, values in enumerate(invalid_cases):
        try:
            P3HumanReview(
                review_item_id=f"review-{index}",
                reviewer_id="user-1",
                **values,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid review case {index} must fail")


def main() -> None:
    test_confirmed_review_has_no_correction_fields()
    test_corrected_review_can_change_event_or_player()
    test_rejected_review_is_explicit_none_not_unreviewed()
    test_invalid_review_states_fail_closed()
    print("p3_review_contract_test: PASS")


if __name__ == "__main__":
    main()
