"use client";

import { useState } from "react";
import type { FormEvent } from "react";
import type { P3ReviewActionsState } from "@/lib/p3/review-actions";
import { formatMetadataTime } from "./review-workspace-types";
import type { SubmitReviewAction, ViewMode } from "./review-workspace-types";
import styles from "./review-workspace.module.css";

type Collection = P3ReviewActionsState["collections"][number];
type Tag = P3ReviewActionsState["tags"][number];

type Props = {
  actions: P3ReviewActionsState;
  view: ViewMode;
  setView: (view: ViewMode) => void;
  totalCount: number;
  newCollectionName: string;
  setNewCollectionName: (value: string) => void;
  newTagName: string;
  setNewTagName: (value: string) => void;
  busy: boolean;
  submit: SubmitReviewAction;
  selectedDerivationId: string | null;
  selectDerivation: (derivationId: string) => Promise<void>;
};

function CollectionManager({
  collection,
  busy,
  submit,
  onDeleted,
}: {
  collection: Collection;
  busy: boolean;
  submit: SubmitReviewAction;
  onDeleted: () => void;
}) {
  const [name, setName] = useState(collection.name);
  const [description, setDescription] = useState(collection.description ?? "");

  async function updateCollection(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    await submit(
      {
        type: "update_collection",
        collection_id: collection.id,
        name: name.trim(),
        description: description.trim() || null,
      },
      `update-collection:${collection.id}`,
      "Collection updated.",
    );
  }

  return (
    <form className={styles.manageForm} onSubmit={updateCollection}>
      <strong>Manage selected collection</strong>
      <input
        value={name}
        maxLength={120}
        aria-label="Collection name"
        onChange={(event) => setName(event.target.value)}
      />
      <textarea
        value={description}
        maxLength={1000}
        aria-label="Collection description"
        placeholder="Optional description"
        onChange={(event) => setDescription(event.target.value)}
      />
      <small>
        {collection.item_count} in this version · {collection.total_item_count} total · updated {formatMetadataTime(collection.updated_at)}
      </small>
      <div className={styles.manageActions}>
        <button type="submit" disabled={busy || !name.trim()}>Save</button>
        <button
          type="button"
          className={styles.dangerButton}
          disabled={busy}
          onClick={async () => {
            const next = await submit(
              { type: "delete_collection", collection_id: collection.id },
              `delete-collection:${collection.id}`,
              "Collection deleted. Review items were not deleted.",
            );
            if (next) onDeleted();
          }}
        >
          Delete
        </button>
      </div>
    </form>
  );
}

function TagManager({
  tag,
  busy,
  submit,
  onDeleted,
}: {
  tag: Tag;
  busy: boolean;
  submit: SubmitReviewAction;
  onDeleted: () => void;
}) {
  const [name, setName] = useState(tag.name);

  async function renameTag(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    await submit(
      { type: "rename_tag", tag_id: tag.id, name: name.trim() },
      `rename-tag:${tag.id}`,
      "Tag updated.",
    );
  }

  return (
    <form className={styles.manageForm} onSubmit={renameTag}>
      <strong>Manage selected tag</strong>
      <input
        value={name}
        maxLength={64}
        aria-label="Tag name"
        onChange={(event) => setName(event.target.value)}
      />
      <small>
        {tag.item_count} in this version · {tag.total_item_count} total · updated {formatMetadataTime(tag.updated_at)}
      </small>
      <div className={styles.manageActions}>
        <button type="submit" disabled={busy || !name.trim()}>Rename</button>
        <button
          type="button"
          className={styles.dangerButton}
          disabled={busy}
          onClick={async () => {
            const next = await submit(
              { type: "delete_tag", tag_id: tag.id },
              `delete-tag:${tag.id}`,
              "Tag deleted. Review items were not deleted.",
            );
            if (next) onDeleted();
          }}
        >
          Delete
        </button>
      </div>
    </form>
  );
}

export function ReviewSidebar(props: Props) {
  const {
    actions,
    view,
    setView,
    totalCount,
    newCollectionName,
    setNewCollectionName,
    newTagName,
    setNewTagName,
    busy,
    submit,
    selectedDerivationId,
    selectDerivation,
  } = props;

  const selectedCollection = view.kind === "collection"
    ? actions.collections.find((collection) => collection.id === view.collectionId) ?? null
    : null;
  const selectedTag = view.kind === "tag"
    ? actions.tags.find((tag) => tag.id === view.tagId) ?? null
    : null;

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
    const next = await submit(
      { type: "create_tag", name },
      "create-tag",
      `Created tag “${name}”.`,
    );
    if (next) setNewTagName("");
  }

  return (
    <aside className={styles.sidebar} aria-label="Review organization">
      <div className={styles.sidebarSection}>
        <div className={styles.sectionTitle}><strong>Review version</strong><small>{actions.available_derivations.length}</small></div>
        <label className={styles.selectLabel} htmlFor="review-derivation">P3 derivation</label>
        <select
          id="review-derivation"
          className={styles.derivationSelect}
          value={selectedDerivationId ?? ""}
          disabled={busy || !actions.available_derivations.length}
          onChange={(event) => void selectDerivation(event.target.value)}
        >
          {selectedDerivationId === null ? <option value="" disabled>{actions.available_derivations.length ? "Select review version" : "No derivation"}</option> : null}
          {actions.available_derivations.map((derivation) => (
            <option key={derivation.id} value={derivation.id}>
              {derivation.status === "current" ? "Current" : derivation.status} · {formatMetadataTime(derivation.generated_at)}
            </option>
          ))}
        </select>
        <p className={styles.metadataText}>
          Version URLs are durable: reopening this URL keeps the selected derivation and organization view.
        </p>
      </div>

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
          <span>Saved</span><small>{actions.saved_review_item_ids.length}</small>
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
            <span>{collection.name}</span><small>{collection.item_count}</small>
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
        {selectedCollection ? (
          <CollectionManager
            key={selectedCollection.id}
            collection={selectedCollection}
            busy={busy}
            submit={submit}
            onDeleted={() => setView({ kind: "all" })}
          />
        ) : null}
      </div>

      <div className={styles.sidebarSection}>
        <div className={styles.sectionTitle}><strong>Tags</strong><small>{actions.tags.length}</small></div>
        <div className={styles.tagNavList}>
          {actions.tags.map((tag) => (
            <button
              key={tag.id}
              type="button"
              className={view.kind === "tag" && view.tagId === tag.id ? styles.tagNavActive : styles.tagNavButton}
              onClick={() => setView({ kind: "tag", tagId: tag.id })}
            >
              <span>{tag.name}</span><small>{tag.item_count}</small>
            </button>
          ))}
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
        {selectedTag ? (
          <TagManager
            key={selectedTag.id}
            tag={selectedTag}
            busy={busy}
            submit={submit}
            onDeleted={() => setView({ kind: "all" })}
          />
        ) : null}
      </div>
    </aside>
  );
}
