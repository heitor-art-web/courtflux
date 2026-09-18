import type { Dispatch, SetStateAction } from "react";
import type { P3ReviewActionsState } from "@/lib/p3/review-actions";
import type { ReviewItem, SubmitReviewAction } from "./review-workspace-types";
import { formatConfidence, formatTime, humanize, reasonLines } from "./review-workspace-types";
import styles from "./review-workspace.module.css";

type Props = {
  item: ReviewItem;
  actions: P3ReviewActionsState;
  saved: boolean;
  busy: boolean;
  noteDrafts: Record<string, string>;
  setNoteDrafts: Dispatch<SetStateAction<Record<string, string>>>;
  submit: SubmitReviewAction;
};

export function ReviewItemCard({ item, actions, saved, busy, noteDrafts, setNoteDrafts, submit }: Props) {
  const note = actions.notes_by_review_item[item.id]?.body ?? "";
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

      <div className={styles.organizeGrid}>
        <details className={styles.organizer} open={Boolean(note)}>
          <summary>Note {note ? "· added" : ""}</summary>
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
                "Note saved.",
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
                    "Note removed.",
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
        <span>{item.completeness === "unknown" ? "Completeness unknown" : humanize(item.completeness)}</span>
      </footer>
    </article>
  );
}
