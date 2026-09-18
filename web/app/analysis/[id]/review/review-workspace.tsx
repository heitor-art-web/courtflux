"use client";

import { useEffect, useMemo, useState } from "react";
import type { P3ReviewActionCommand, P3ReviewActionsState } from "@/lib/p3/review-actions";
import { ReviewItemCard } from "./review-item-card";
import { ReviewSidebar } from "./review-sidebar";
import type { ThingsToReviewResponse, ViewMode } from "./review-workspace-types";
import styles from "./review-workspace.module.css";

export function ReviewWorkspace({ runId }: { runId: string }) {
  const [review, setReview] = useState<ThingsToReviewResponse | null>(null);
  const [actions, setActions] = useState<P3ReviewActionsState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>({ kind: "all" });
  const [noteDrafts, setNoteDrafts] = useState<Record<string, string>>({});
  const [newTagName, setNewTagName] = useState("");
  const [newCollectionName, setNewCollectionName] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      setLoading(true);
      setError("");
      try {
        const [reviewResponse, actionResponse] = await Promise.all([
          fetch(`/api/analyses/${encodeURIComponent(runId)}/things-to-review`, { cache: "no-store", signal: controller.signal }),
          fetch(`/api/analyses/${encodeURIComponent(runId)}/review-actions`, { cache: "no-store", signal: controller.signal }),
        ]);
        const reviewPayload = (await reviewResponse.json()) as ThingsToReviewResponse;
        const actionPayload = (await actionResponse.json()) as P3ReviewActionsState & { error?: string };
        if (!reviewResponse.ok) throw new Error(reviewPayload.error || `Could not load review items (HTTP ${reviewResponse.status})`);
        if (!actionResponse.ok) throw new Error(actionPayload.error || `Could not load review actions (HTTP ${actionResponse.status})`);
        setReview(reviewPayload);
        setActions(actionPayload);
        setNoteDrafts(Object.fromEntries(
          Object.entries(actionPayload.notes_by_review_item).map(([id, note]) => [id, (note as { body: string }).body]),
        ));
      } catch (loadError) {
        if (!controller.signal.aborted) setError(loadError instanceof Error ? loadError.message : String(loadError));
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [runId]);

  async function submit(command: P3ReviewActionCommand, key: string, successMessage: string) {
    setBusyKey(key);
    setStatusMessage("");
    setError("");
    try {
      const response = await fetch(`/api/analyses/${encodeURIComponent(runId)}/review-actions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(command),
      });
      const payload = (await response.json()) as P3ReviewActionsState & { error?: string };
      if (!response.ok) throw new Error(payload.error || `Review action failed (HTTP ${response.status})`);
      setActions(payload);
      setStatusMessage(successMessage);
      return payload;
    } catch (actionError) {
      setError(actionError instanceof Error ? actionError.message : String(actionError));
      return null;
    } finally {
      setBusyKey(null);
    }
  }

  const savedIds = useMemo(() => new Set(actions?.saved_review_item_ids ?? []), [actions?.saved_review_item_ids]);
  const currentCollectionCounts = useMemo(() => {
    const counts = new Map<string, number>();
    if (!actions || !review) return counts;
    for (const item of review.items) {
      for (const collectionId of actions.collection_ids_by_review_item[item.id] ?? []) {
        counts.set(collectionId, (counts.get(collectionId) ?? 0) + 1);
      }
    }
    return counts;
  }, [actions, review]);
  const visibleItems = useMemo(() => {
    const items = review?.items ?? [];
    if (!actions) return items;
    if (view.kind === "saved") return items.filter((item) => savedIds.has(item.id));
    if (view.kind === "collection") {
      return items.filter((item) => (actions.collection_ids_by_review_item[item.id] ?? []).includes(view.collectionId));
    }
    return items;
  }, [actions, review?.items, savedIds, view]);

  if (loading) return <div className={styles.stateBox}>Loading review workspace…</div>;
  if (error && (!review || !actions)) return <div className={styles.errorBox} role="alert">{error}</div>;
  if (!review || !actions) return <div className={styles.stateBox}>Review workspace is unavailable for this analysis.</div>;

  const currentCollection = view.kind === "collection"
    ? actions.collections.find((collection) => collection.id === view.collectionId) ?? null
    : null;

  return (
    <section className={styles.workspace} aria-labelledby="review-workspace-title">
      <header className={styles.header}>
        <div>
          <div className={styles.eyebrow}>P3 review actions</div>
          <h1 id="review-workspace-title">Organize what matters</h1>
          <p>Save moments, add notes and group review items without changing the underlying CourtFlux prediction.</p>
        </div>
        <div className={styles.summaryGrid} aria-label="Review action summary">
          <div><strong>{review.summary.total_items}</strong><span>review items</span></div>
          <div><strong>{savedIds.size}</strong><span>saved</span></div>
          <div><strong>{actions.collections.length}</strong><span>collections</span></div>
        </div>
      </header>

      {!review.derivation ? (
        <div className={styles.stateBox}>
          <strong>No P3 review derivation yet.</strong>
          <span>The organization UI is ready, but this run has no materialized Things to Review items yet.</span>
        </div>
      ) : null}
      {error ? <div className={styles.errorBox} role="alert">{error}</div> : null}
      {statusMessage ? <div className={styles.successBox} role="status">{statusMessage}</div> : null}

      <div className={styles.layout}>
        <ReviewSidebar
          actions={actions}
          view={view}
          setView={setView}
          savedCount={savedIds.size}
          totalCount={review.items.length}
          currentCollectionCounts={currentCollectionCounts}
          newCollectionName={newCollectionName}
          setNewCollectionName={setNewCollectionName}
          newTagName={newTagName}
          setNewTagName={setNewTagName}
          busy={busyKey !== null}
          submit={submit}
        />

        <div className={styles.mainColumn}>
          <div className={styles.viewHeader}>
            <div>
              <strong>{view.kind === "all" ? "All review items" : view.kind === "saved" ? "Saved items" : currentCollection?.name ?? "Collection"}</strong>
              <span>{visibleItems.length} in this view · chronological order · no ranking applied</span>
            </div>
          </div>
          {!visibleItems.length ? (
            <div className={styles.stateBox}>
              <strong>Nothing here yet.</strong>
              <span>{view.kind === "all" ? "This run has no materialized review items." : "Organize an item into this view to see it here."}</span>
            </div>
          ) : null}
          <div className={styles.itemList}>
            {visibleItems.map((item) => (
              <ReviewItemCard
                key={item.id}
                item={item}
                actions={actions}
                saved={savedIds.has(item.id)}
                busy={busyKey !== null}
                noteDrafts={noteDrafts}
                setNoteDrafts={setNoteDrafts}
                submit={submit}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
