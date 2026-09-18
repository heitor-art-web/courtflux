"use client";

import { useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { HUMAN_REVIEW_EVENT_TYPES } from "@/lib/p3/human-review";
import type { HumanReviewEventType } from "@/lib/p3/human-review";
import type { P3ReviewActionsState } from "@/lib/p3/review-actions";
import type { ReviewItem, SubmitHumanReview, SubmitReviewAction } from "./review-workspace-types";
import { formatConfidence, formatMetadataTime, formatTime, humanize, reasonLines } from "./review-workspace-types";
import styles from "./review-workspace.module.css";
import reviewStyles from "./human-review.module.css";

type Props = {
  item: ReviewItem;
  actions: P3ReviewActionsState;
  saved: boolean;
  savedAt: string | null;
  busy: boolean;
  noteDrafts: Record<string, string>;
  setNoteDrafts: Dispatch<SetStateAction<Record<string, string>>>;
  submit: SubmitReviewAction;
  submitHumanReview: SubmitHumanReview;
  availableGamePlayerIds: string[];
};

function HumanReviewEditor({
  item,
  busy,
  submitHumanReview,
  availableGamePlayerIds,
}: {
  item: ReviewItem;
  busy: boolean;
  submitHumanReview: SubmitHumanReview;
  availableGamePlayerIds: string[];
}) {
  const review = item.my_review;
  const initialEvent = review?.state === "corrected"
    && review.corrected_event_type
    && review.corrected_event_type !== "none"
    ? review.corrected_event_type as HumanReviewEventType
    : "";
  const [eventType, setEventType] = useState<HumanReviewEventType | "">(initialEvent);
  const [playerId, setPlayerId] = useState(review?.corrected_game_player_id ?? "");
  const [rationale, setRationale] = useState(review?.note ?? "");
  const canCorrect = Boolean(eventType || playerId);

  const reviewSummary = review?.state === "confirmed"
    ? "Confirmed"
    : review?.state === "rejected"
      ? "Rejected · NONE"
      : review?.state === "corrected"
        ? [
            "Corrected",
            review.corrected_event_type && review.corrected_event_type !== "none"
              ? humanize(review.corrected_event_type)
              : null,
            review.corrected_game_player_id ? humanize(review.corrected_game_player_id) : null,
          ].filter(Boolean).join(" · ")
        : "Unreviewed";

  return (
    <section className={reviewStyles.humanReviewBox} aria-label={`Human review for ${humanize(item.review_type)}`}>
      <div className={reviewStyles.humanReviewHeader}>
        <div>
          <strong>Human review</strong>
          <span>
            Original prediction stays {humanize(item.review_type)} · {formatConfidence(item.confidence)}
          </span>
        </div>
        <div className={reviewStyles.reviewStateBadge} data-state={review?.state ?? "unreviewed"}>
          {reviewSummary}
        </div>
      </div>

      {review ? (
        <p className={reviewStyles.reviewMetadata}>
          Reviewed {formatMetadataTime(review.updated_at)}. Human truth is stored separately from the model prediction.
        </p>
      ) : (
        <p className={reviewStyles.reviewMetadata}>
          Leaving this item untouched remains distinct from rejecting it as NONE.
        </p>
      )}

      <label className={reviewStyles.reviewNoteLabel}>
        Review rationale <span>optional</span>
        <textarea
          className={reviewStyles.reviewNote}
          value={rationale}
          maxLength={2000}
          disabled={busy}
          onChange={(event) => setRationale(event.target.value)}
          placeholder="Optional note about why you confirmed or corrected this moment"
        />
      </label>

      <div className={reviewStyles.reviewDecisionRow}>
        <button
          type="button"
          disabled={busy}
          onClick={() => void submitHumanReview(
            {
              type: "confirm",
              review_item_id: item.id,
              note: rationale.trim() || null,
            },
            `human-confirm:${item.id}`,
            "Prediction confirmed.",
          )}
        >
          Confirm prediction
        </button>
        <button
          type="button"
          className={reviewStyles.rejectButton}
          disabled={busy}
          onClick={() => void submitHumanReview(
            {
              type: "reject",
              review_item_id: item.id,
              note: rationale.trim() || null,
            },
            `human-reject:${item.id}`,
            "Marked as NONE. The original prediction was preserved.",
          )}
        >
          NONE · reject event
        </button>
      </div>

      <details className={reviewStyles.correctionPanel} open={review?.state === "corrected"}>
        <summary>Correct action or player</summary>
        <div className={reviewStyles.correctionGrid}>
          <label>
            Correct action
            <select
              value={eventType}
              disabled={busy}
              onChange={(event) => setEventType(event.target.value as HumanReviewEventType | "")}
            >
              <option value="">Keep action unchanged</option>
              {HUMAN_REVIEW_EVENT_TYPES.map((value) => (
                <option key={value} value={value}>{humanize(value)}</option>
              ))}
            </select>
          </label>

          <label>
            Correct game player
            <select
              value={playerId}
              disabled={busy || !availableGamePlayerIds.length}
              onChange={(event) => setPlayerId(event.target.value)}
            >
              <option value="">
                {availableGamePlayerIds.length ? "Keep player attribution unchanged" : "No known game-player IDs"}
              </option>
              {availableGamePlayerIds.map((value) => (
                <option key={value} value={value}>{humanize(value)}</option>
              ))}
            </select>
          </label>
        </div>
        <div className={reviewStyles.correctionActions}>
          <button
            type="button"
            disabled={busy || !canCorrect}
            onClick={() => void submitHumanReview(
              {
                type: "correct",
                review_item_id: item.id,
                corrected_event_type: eventType || null,
                corrected_game_player_id: playerId || null,
                note: rationale.trim() || null,
              },
              `human-correct:${item.id}`,
              "Human correction saved. Original prediction preserved.",
            )}
          >
            Save correction
          </button>
          {!canCorrect ? <span>Select an action or player correction.</span> : null}
        </div>
      </details>
    </section>
  );
}

export function ReviewItemCard({
  item,
  actions,
  saved,
  savedAt,
  busy,
  noteDrafts,
  setNoteDrafts,
  submit,
  submitHumanReview,
  availableGamePlayerIds,
}: Props) {
  const noteMetadata = actions.notes_by_review_item[item.id] ?? null;
  const note = noteMetadata?.body ?? "";
  const draft = noteDrafts[item.id] ?? note;
  const itemTagIds = new Set(actions.tag_ids_by_review_item[item.id] ?? []);
  const itemCollectionIds = new Set(actions.collection_ids_by_review_item[item.id] ?? []);
  const reasons = reasonLines(item.reason_shown);

  return (
    <article className={styles.reviewCard}>
      <div className={styles.cardHeader}>
        <div>
          <div className={styles.cardMeta}>
            <span>{formatTime(item.source_time.start_ms)}</span>
            <span>{humanize(item.support_state)}</span>
            <span>{formatConfidence(item.confidence)}</span>
          </div>
          <h2>{humanize(item.review_type)}</h2>
          <div className={styles.identityLine}>
            {item.game_player_ids.length ? item.game_player_ids.map(humanize).join(" · ") : "Player unknown"}
            {item.team_id ? ` · ${humanize(item.team_id)}` : ""}
          </div>
        </div>
        <button
          className={saved ? styles.savedButtonActive : styles.savedButton}
          type="button"
          disabled={busy}
          aria-pressed={saved}
          onClick={() => void submit(
            { type: saved ? "unsave" : "save", review_item_id: item.id },
            `save:${item.id}`,
            saved ? "Removed from Saved." : "Saved for later review.",
          )}
        >
          {saved ? "Saved" : "Save"}
        </button>
      </div>

      {item.clip?.artifact?.playable_url ? (
        <video className={styles.clip} controls preload="metadata" src={item.clip.artifact.playable_url}>
          Your browser does not support video playback.
        </video>
      ) : (
        <div className={styles.clipState}>
          Clip {item.clip?.status ?? "unavailable"}{item.clip?.error ? ` · ${item.clip.error}` : ""}
        </div>
      )}

      {reasons.length ? (
        <div className={styles.reasonBox}>
          <strong>Why shown</strong>
          <ul>{reasons.map((reason, index) => <li key={`${index}-${reason}`}>{reason}</li>)}</ul>
        </div>
      ) : null}

      <HumanReviewEditor
        key={`${item.id}:${item.my_review?.updated_at ?? "unreviewed"}`}
        item={item}
        busy={busy}
        submitHumanReview={submitHumanReview}
        availableGamePlayerIds={availableGamePlayerIds}
      />

      <div className={styles.organizeGrid}>
        <details className={styles.organizer} open={Boolean(note)}>
          <summary>Personal note {noteMetadata ? `· updated ${formatMetadataTime(noteMetadata.updated_at)}` : ""}</summary>
          <textarea
            value={draft}
            maxLength={4000}
            onChange={(event) => setNoteDrafts((current) => ({ ...current, [item.id]: event.target.value }))}
            placeholder="What should you remember about this moment?"
          />
          <div className={styles.inlineActions}>
            <button
              type="button"
              disabled={busy || !draft.trim() || draft.trim() === note}
              onClick={() => void submit(
                { type: "set_note", review_item_id: item.id, body: draft.trim() },
                `note:${item.id}`,
                "Personal note saved.",
              )}
            >
              Save note
            </button>
            {note ? (
              <button
                type="button"
                className={styles.subtleButton}
                disabled={busy}
                onClick={async () => {
                  const next = await submit(
                    { type: "delete_note", review_item_id: item.id },
                    `delete-note:${item.id}`,
                    "Personal note removed.",
                  );
                  if (next) setNoteDrafts((current) => ({ ...current, [item.id]: "" }));
                }}
              >
                Delete
              </button>
            ) : null}
          </div>
        </details>

        <details className={styles.organizer}>
          <summary>Tags · {itemTagIds.size}</summary>
          <div className={styles.optionList}>
            {actions.tags.map((tag) => {
              const active = itemTagIds.has(tag.id);
              return (
                <button
                  key={tag.id}
                  type="button"
                  aria-pressed={active}
                  className={active ? styles.optionActive : styles.optionButton}
                  disabled={busy}
                  onClick={() => void submit(
                    { type: active ? "untag" : "tag", review_item_id: item.id, tag_id: tag.id },
                    `tag:${item.id}:${tag.id}`,
                    active ? `Removed tag “${tag.name}”.` : `Added tag “${tag.name}”.`,
                  )}
                >
                  {active ? "✓ " : "+ "}{tag.name}
                </button>
              );
            })}
            {!actions.tags.length ? <span className={styles.muted}>Create a tag from the sidebar first.</span> : null}
          </div>
        </details>

        <details className={styles.organizer}>
          <summary>Collections · {itemCollectionIds.size}</summary>
          <div className={styles.optionList}>
            {actions.collections.map((collection) => {
              const active = itemCollectionIds.has(collection.id);
              return (
                <button
                  key={collection.id}
                  type="button"
                  aria-pressed={active}
                  className={active ? styles.optionActive : styles.optionButton}
                  disabled={busy}
                  onClick={() => void submit(
                    {
                      type: active ? "remove_from_collection" : "add_to_collection",
                      review_item_id: item.id,
                      collection_id: collection.id,
                    },
                    `collection:${item.id}:${collection.id}`,
                    active ? `Removed from “${collection.name}”.` : `Added to “${collection.name}”.`,
                  )}
                >
                  {active ? "✓ " : "+ "}{collection.name}
                </button>
              );
            })}
            {!actions.collections.length ? <span className={styles.muted}>Create a collection from the sidebar first.</span> : null}
          </div>
        </details>
      </div>

      <footer className={styles.cardFooter}>
        <span>{item.my_review ? `Human review: ${humanize(item.my_review.state)}` : "Not human-reviewed"}</span>
        <span>{savedAt ? `Saved ${formatMetadataTime(savedAt)}` : item.completeness === "unknown" ? "Completeness unknown" : humanize(item.completeness)}</span>
      </footer>
    </article>
  );
}
