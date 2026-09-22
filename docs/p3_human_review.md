# P3.5 — Human Review + Correction

**Status:** complete for P3 foundation development.

P3.5 gives an authenticated reviewer a first-class way to confirm or correct a durable P3 review item while preserving the original CourtFlux prediction, confidence, source evidence and provenance.

## Review states

The durable human states are:

```text
UNREVIEWED  = no p3_reviews row exists for this reviewer/item
CONFIRMED   = prediction accepted as shown
CORRECTED   = action and/or game-player attribution changed by the reviewer
REJECTED    = explicit NONE; reviewer says no relevant correction-taxonomy event occurred
```

`UNREVIEWED` and `REJECTED/NONE` are intentionally different. Simply skipping an item never creates ground truth.

## Correction taxonomy

P3.5 supports the initial correction labels:

```text
PASS
TURNOVER
SHOT
REBOUND
LOOSE_BALL
DUNK
ALLEY_OOP
NONE
```

`NONE` is represented only through `review_state = rejected`. It is never accepted as a `corrected` action.

## Immutable prediction boundary

Human review writes only to `p3_reviews`.

It does not update:

- `p3_review_items.review_type`;
- original confidence/support state;
- source event IDs;
- source-event provenance;
- clip/source timestamps;
- original game-player attribution.

The read model returns prediction and human truth side by side.

## Player correction

A reviewer may replace game-level attribution using `corrected_game_player_id`.

The API accepts only game-player IDs observed in the selected P3 derivation's normalized source events. This prevents arbitrary identifiers while keeping P3 independent from persistent roster identity, which remains a later phase.

## API

The command surface is:

```text
POST /api/analyses/<analysis_run_id>/human-review
POST /api/analyses/<analysis_run_id>/human-review?derivation_id=<historical_derivation_id>
```

Commands:

```json
{ "type": "confirm", "review_item_id": "...", "note": null }
```

```json
{
  "type": "correct",
  "review_item_id": "...",
  "corrected_event_type": "turnover",
  "corrected_game_player_id": "game_player_05",
  "note": "optional rationale"
}
```

```json
{ "type": "reject", "review_item_id": "...", "note": "not a supported event" }
```

The authenticated Supabase client performs the upsert so the existing `p3_reviews` RLS policy remains the final permission authority. The service-role client is used only for read-side context validation.

## UI

Each Things to Review card now exposes:

- original predicted action;
- original confidence;
- current human-review state;
- Confirm prediction;
- Correct action;
- Correct game player;
- explicit `NONE · reject event`;
- optional review rationale;
- review timestamp.

Reviewed and Unreviewed are dedicated review views and remain chronological. P3.5 introduces no ranking or confidence threshold.

## Feedback-to-model boundary

The resulting data can later support:

- evaluation datasets;
- error analysis;
- calibration/threshold work;
- heuristic/model improvements;
- controlled future training or fine-tuning.

P3.5 does **not** perform online learning, per-user model mutation or automatic promotion of human corrections into production predictions.

## P3.5 exit checklist

- [x] Confirm is durable;
- [x] Correct action is durable;
- [x] Correct game player is durable;
- [x] explicit NONE maps to rejected;
- [x] unreviewed remains distinct from rejected;
- [x] original model prediction remains immutable;
- [x] review state and prediction are returned separately;
- [x] corrections work on current and historical P3 derivations;
- [x] corrected player IDs are constrained to known game-player IDs in that derivation;
- [x] authenticated RLS remains the write authority;
- [x] reviewed/unreviewed views exist in the product surface;
- [x] no ranking or P2-accuracy threshold is introduced;
- [x] no automatic online learning is introduced.

P3.5 completion does not assert that P2 predictions are accurate enough for final P3 product validation. It guarantees that humans can create durable, auditable truth labels against preserved model evidence.
