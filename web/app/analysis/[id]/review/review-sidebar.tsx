import type { FormEvent } from "react";
import type { P3ReviewActionsState } from "@/lib/p3/review-actions";
import type { SubmitReviewAction, ViewMode } from "./review-workspace-types";
import styles from "./review-workspace.module.css";

type Props = {
  actions: P3ReviewActionsState;
  view: ViewMode;
  setView: (view: ViewMode) => void;
  savedCount: number;
  totalCount: number;
  currentCollectionCounts: Map<string, number>;
  newCollectionName: string;
  setNewCollectionName: (value: string) => void;
  newTagName: string;
  setNewTagName: (value: string) => void;
  busy: boolean;
  submit: SubmitReviewAction;
};

export function ReviewSidebar(props: Props) {
  const {
    actions,
    view,
    setView,
    savedCount,
    totalCount,
    currentCollectionCounts,
    newCollectionName,
    setNewCollectionName,
    newTagName,
    setNewTagName,
    busy,
    submit,
  } = props;

  async function createCollection(event: FormEvent) {
    event.preventDefault();
    const name = newCollectionName.trim();
    if (!name) return;
    const next = await submit(
      { type: "create_collection", name },
      "create-collection",
      `Created collection “${name}”.`,
    );
    if (next) setNewCollectionName("");
  }

  async function createTag(event: FormEvent) {
    event.preventDefault();
    const name = newTagName.trim();
    if (!name) return;
    const next = await submit({ type: "create_tag", name }, "create-tag", `Created tag “${name}”.`);
    if (next) setNewTagName("");
  }

  return (
    <aside className={styles.sidebar} aria-label="Review organization">
      <div className={styles.sidebarSection}>
        <strong>Views</strong>
        <button
          className={view.kind === "all" ? styles.navActive : styles.navButton}
          type="button"
          onClick={() => setView({ kind: "all" })}
        >
          <span>All review items</span><small>{totalCount}</small>
        </button>
        <button
          className={view.kind === "saved" ? styles.navActive : styles.navButton}
          type="button"
          onClick={() => setView({ kind: "saved" })}
        >
          <span>Saved</span><small>{savedCount}</small>
        </button>
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.sectionTitle}><strong>Collections</strong><small>{actions.collections.length}</small></div>
        {actions.collections.map((collection) => (
          <button
            key={collection.id}
            className={view.kind === "collection" && view.collectionId === collection.id ? styles.navActive : styles.navButton}
            type="button"
            onClick={() => setView({ kind: "collection", collectionId: collection.id })}
          >
            <span>{collection.name}</span><small>{currentCollectionCounts.get(collection.id) ?? 0}</small>
          </button>
        ))}
        {!actions.collections.length ? <p className={styles.muted}>No collections yet.</p> : null}
        <form className={styles.createForm} onSubmit={createCollection}>
          <label htmlFor="new-collection">New collection</label>
          <div className={styles.inputRow}>
            <input
              id="new-collection"
              value={newCollectionName}
              maxLength={120}
              onChange={(event) => setNewCollectionName(event.target.value)}
              placeholder="e.g. Turnovers to revisit"
            />
            <button type="submit" disabled={!newCollectionName.trim() || busy}>Add</button>
          </div>
        </form>
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.sectionTitle}><strong>Tags</strong><small>{actions.tags.length}</small></div>
        <div className={styles.tagCloud}>
          {actions.tags.map((tag) => <span key={tag.id}>{tag.name}</span>)}
          {!actions.tags.length ? <span className={styles.muted}>No tags yet.</span> : null}
        </div>
        <form className={styles.createForm} onSubmit={createTag}>
          <label htmlFor="new-tag">New tag</label>
          <div className={styles.inputRow}>
            <input
              id="new-tag"
              value={newTagName}
              maxLength={64}
              onChange={(event) => setNewTagName(event.target.value)}
              placeholder="e.g. transition"
            />
            <button type="submit" disabled={!newTagName.trim() || busy}>Add</button>
          </div>
        </form>
      </div>
    </aside>
  );
}
