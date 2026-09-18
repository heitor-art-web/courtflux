"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { P3ReviewActionCommand, P3ReviewActionsState } from "@/lib/p3/review-actions";
import { ReviewItemCard } from "./review-item-card";
import { ReviewSidebar } from "./review-sidebar";
import type { ThingsToReviewResponse, ViewMode } from "./review-workspace-types";
import styles from "./review-workspace.module.css";

type Props = {
  runId: string;
  initialDerivationId: string | null;
  initialView: ViewMode;
};

function endpoint(runId: string, resource: "things-to-review" | "review-actions", derivationId: string | null) {
  const base = `/api/analyses/${encodeURIComponent(runId)}/${resource}`;
  return derivationId ? `${base}?derivation_id=${encodeURIComponent(derivationId)}` : base;
}


function availableView(view: ViewMode, actions: P3ReviewActionsState): ViewMode {
  if (view.kind === "collection" && !actions.collections.some((item) => item.id === view.collectionId)) {
    return { kind: "all" };
  }
  if (view.kind === "tag" && !actions.tags.some((item) => item.id === view.tagId)) {
    return { kind: "all" };
  }
  return view;
}

function syncUrl(derivationId: string | null, view: ViewMode) {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  if (derivationId) url.searchParams.set("derivation", derivationId);
  else url.searchParams.delete("derivation");
  url.searchParams.delete("view");
  url.searchParams.delete("collection");
  url.searchParams.delete("tag");
  if (view.kind === "saved") url.searchParams.set("view", "saved");
  if (view.kind === "collection") {
    url.searchParams.set("view", "collection");
    url.searchParams.set("collection", view.collectionId);
  }
  if (view.kind === "tag") {
    url.searchParams.set("view", "tag");
    url.searchParams.set("tag", view.tagId);
  }
  window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
}

export function ReviewWorkspace({ runId, initialDerivationId, initialView }: Props) {
  const [review, setReview] = useState<ThingsToReviewResponse | null>(null);
  const [actions, setActions] = useState<P3ReviewActionsState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [view, setViewState] = useState<ViewMode>(initialView);
  const [derivationId, setDerivationId] = useState<string | null>(initialDerivationId);
  const [noteDrafts, setNoteDrafts] = useState<Record<string, string>>({});
  const [newTagName, setNewTagName] = useState("");
  const [newCollectionName, setNewCollectionName] = useState("");

  const load = useCallback(async (requestedDerivation: string | null, signal?: AbortSignal) => {
    const [reviewResponse, actionResponse] = await Promise.all([
      fetch(endpoint(runId, "things-to-review", requestedDerivation), { cache: "no-store", signal }),
      fetch(endpoint(runId, "review-actions", requestedDerivation), { cache: "no-store", signal }),
    ]);
    const reviewPayload = (await reviewResponse.json()) as ThingsToReviewResponse;
    const actionPayload = (await actionResponse.json()) as P3ReviewActionsState & { error?: string };
    if (!reviewResponse.ok) {
      throw new Error(reviewPayload.error || `Could not load review items (HTTP ${reviewResponse.status})`);
    }
    if (!actionResponse.ok) {
      throw new Error(actionPayload.error || `Could not load review actions (HTTP ${actionResponse.status})`);
    }
    return { reviewPayload, actionPayload };
  }, [runId]);

  useEffect(() => {
    const controller = new AbortController();
    async function boot() {
      setLoading(true);
      setError("");
      try {
        const { reviewPayload, actionPayload } = await load(initialDerivationId, controller.signal);
        if (controller.signal.aborted) return;
        const resolvedDerivation = actionPayload.derivation_id ?? reviewPayload.derivation?.id ?? null;
        const resolvedView = availableView(initialView, actionPayload);
        setReview(reviewPayload);
        setActions(actionPayload);
        setViewState(resolvedView);
        setDerivationId(resolvedDerivation);
        setNoteDrafts(Object.fromEntries(
          Object.entries(actionPayload.notes_by_review_item).map(([id, note]) => [id, note.body]),
        ));
        syncUrl(resolvedDerivation, resolvedView);
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(loadError instanceof Error ? loadError.message : String(loadError));
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void boot();
    return () => controller.abort();
  }, [initialDerivationId, initialView, load, runId]);

  function setView(next: ViewMode) {
    setViewState(next);
    syncUrl(derivationId, next);
  }

  async function selectDerivation(nextDerivationId: string) {
    if (nextDerivationId === derivationId) return;
    setBusyKey("derivation");
    setStatusMessage("");
    setError("");
    try {
      const { reviewPayload, actionPayload } = await load(nextDerivationId);
      setReview(reviewPayload);
      setActions(actionPayload);
      setDerivationId(actionPayload.derivation_id ?? nextDerivationId);
      setNoteDrafts(Object.fromEntries(
        Object.entries(actionPayload.notes_by_review_item).map(([id, note]) => [id, note.body]),
      ));
      syncUrl(actionPayload.derivation_id ?? nextDerivationId, view);
      setStatusMessage(
        actionPayload.derivation_status === "current"
          ? "Opened current P3 review version."
          : "Opened historical P3 review version.",
      );
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : String(loadError));
    } finally {
      setBusyKey(null);
    }
  }

  async function submit(command: P3ReviewActionCommand, key: string, successMessage: string) {
    setBusyKey(key);
    setStatusMessage("");
    setError("");
    try {
      const response = await fetch(endpoint(runId, "review-actions", derivationId), {
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

  const savedIds = useMemo(
    () => new Set(actions?.saved_review_item_ids ?? []),
    [actions?.saved_review_item_ids],
  );
  const savedAtById = useMemo(
    () => new Map((actions?.saved_items ?? []).map((entry) => [entry.review_item_id, entry.created_at])),
    [actions?.saved_items],
  );

  const visibleItems = useMemo(() => {
    const items = review?.items ?? [];
    if (!actions) return items;
    if (view.kind === "saved") return items.filter((item) => savedIds.has(item.id));
    if (view.kind === "collection") {
      return items.filter((item) =>
        (actions.collection_ids_by_review_item[item.id] ?? []).includes(view.collectionId),
      );
    }
    if (view.kind === "tag") {
      return items.filter((item) =>
        (actions.tag_ids_by_review_item[item.id] ?? []).includes(view.tagId),
      );
    }
    return items;
  }, [actions, review?.items, savedIds, view]);

  if (loading) return <div className={styles.stateBox}>Loading review workspace…</div>;
  if (error && (!review || !actions)) return <div className={styles.errorBox} role="alert">{error}</div>;
  if (!review || !actions) {
    return <div className={styles.stateBox}>Review workspace is unavailable for this analysis.</div>;
  }

  const currentCollection = view.kind === "collection"
    ? actions.collections.find((collection) => collection.id === view.collectionId) ?? null
    : null;
  const currentTag = view.kind === "tag"
    ? actions.tags.find((tag) => tag.id === view.tagId) ?? null
    : null;
  const viewTitle = view.kind === "all"
    ? "All review items"
    : view.kind === "saved"
      ? "Saved items"
      : view.kind === "tag"
        ? currentTag?.name ?? "Tag"
        : currentCollection?.name ?? "Collection";

  return (
    <section className={styles.workspace} aria-labelledby="review-workspace-title">
      <header className={styles.header}>
        <div>
          <div className={styles.eyebrow}>P3 review actions</div>
          <h1 id="review-workspace-title">Organize what matters</h1>
          <p>
            Save moments, add personal notes, tags and collections without changing the underlying CourtFlux prediction.
          </p>
        </div>
        <div className={styles.summaryGrid} aria-label="Review action summary">
          <div><strong>{review.summary.total_items}</strong><span>review items</span></div>
          <div><strong>{savedIds.size}</strong><span>saved in version</span></div>
          <div><strong>{actions.collections.length}</strong><span>collections</span></div>
        </div>
      </header>

      {!review.derivation ? (
        <div className={styles.stateBox}>
          <strong>No P3 review derivation yet.</strong>
          <span>The organization UI is ready, but this run has no materialized Things to Review items yet.</span>
        </div>
      ) : null}
      {actions.derivation_status && actions.derivation_status !== "current" ? (
        <div className={styles.historyBox} role="status">
          <strong>Historical review version</strong>
          <span>
            You are reopening a {actions.derivation_status} P3 derivation. Your personal organization remains editable and the original evidence stays unchanged.
          </span>
        </div>
      ) : null}
      {error ? <div className={styles.errorBox} role="alert">{error}</div> : null}
      {statusMessage ? <div className={styles.successBox} role="status">{statusMessage}</div> : null}

      <div className={styles.layout}>
        <ReviewSidebar
          actions={actions}
          view={view}
          setView={setView}
          totalCount={review.items.length}
          newCollectionName={newCollectionName}
          setNewCollectionName={setNewCollectionName}
          newTagName={newTagName}
          setNewTagName={setNewTagName}
          busy={busyKey !== null}
          submit={submit}
          selectedDerivationId={derivationId}
          selectDerivation={selectDerivation}
        />

        <div className={styles.mainColumn}>
          <div className={styles.viewHeader}>
            <div>
              <strong>{viewTitle}</strong>
              <span>
                {visibleItems.length} in this view · chronological order · no ranking applied
                {review.derivation ? ` · ${review.derivation.status} derivation` : ""}
              </span>
              {currentCollection?.description ? <p>{currentCollection.description}</p> : null}
            </div>
          </div>
          {!visibleItems.length ? (
            <div className={styles.stateBox}>
              <strong>Nothing here yet.</strong>
              <span>
                {view.kind === "all"
                  ? "This review version has no materialized review items."
                  : "Organize an item into this view to see it here."}
              </span>
            </div>
          ) : null}
          <div className={styles.itemList}>
            {visibleItems.map((item) => (
              <ReviewItemCard
                key={item.id}
                item={item}
                actions={actions}
                saved={savedIds.has(item.id)}
                savedAt={savedAtById.get(item.id) ?? null}
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
