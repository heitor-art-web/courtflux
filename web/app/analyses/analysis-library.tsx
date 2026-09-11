"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import styles from "./page.module.css";

export type LibraryAnalysis = {
  id: string;
  status: string;
  title: string;
  competition: string | null;
  gameDate: string | null;
  createdAt: string;
  completedAt: string | null;
  durationMinutes: number | null;
  sourceKind: "upload" | "youtube" | "other";
  sourceExpiresAt: string | null;
  sourceDeletedAt: string | null;
  sourceRetentionKnown: boolean;
  searchText: string;
};

type StatusFilter = "all" | "completed" | "processing" | "failed";
type SourceFilter = "all" | LibraryAnalysis["sourceKind"];

function formatDate(value: string | null | undefined) {
  if (!value) return "Date not provided";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Date not provided";
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

function formatMinutes(value: number | null) {
  if (value === null || !Number.isFinite(value)) return null;
  if (value < 10) return `${value.toFixed(1).replace(/\.0$/, "")} min`;
  return `${Math.round(value)} min`;
}

function statusGroup(status: string): Exclude<StatusFilter, "all"> {
  if (status === "completed") return "completed";
  if (status === "queued" || status === "running") return "processing";
  return "failed";
}

function statusLabel(status: string) {
  if (status === "running") return "Processing";
  if (status === "queued") return "Queued";
  if (status === "completed") return "Completed";
  if (status === "canceled") return "Canceled";
  if (status === "failed") return "Failed";
  return status;
}

function statusClass(status: string) {
  if (status === "completed") return `${styles.status} ${styles.statusCompleted}`;
  if (status === "running") return `${styles.status} ${styles.statusRunning}`;
  if (status === "queued") return `${styles.status} ${styles.statusQueued}`;
  return `${styles.status} ${styles.statusFailed}`;
}

function sourceLabel(kind: LibraryAnalysis["sourceKind"]) {
  if (kind === "upload") return "Uploaded video";
  if (kind === "youtube") return "YouTube";
  return "Other source";
}

function retentionCopy(analysis: LibraryAnalysis) {
  if (analysis.sourceKind === "youtube") {
    return { label: "YouTube source · Results saved in CourtFlux", className: styles.retentionNeutral };
  }
  if (analysis.sourceKind !== "upload") {
    return { label: "Results saved in CourtFlux", className: styles.retentionNeutral };
  }
  if (analysis.sourceDeletedAt) {
    return { label: "Source video no longer stored · Results preserved", className: styles.retentionPreserved };
  }
  if (analysis.sourceExpiresAt) {
    return {
      label: `Source video stored until ${formatDate(analysis.sourceExpiresAt)} · Results remain afterward`,
      className: styles.retentionAvailable,
    };
  }
  return {
    label: analysis.sourceRetentionKnown ? "Uploaded source follows CourtFlux retention policy" : "Uploaded source · Retention status unavailable",
    className: styles.retentionNeutral,
  };
}

export function AnalysisLibrary({ analyses, totalCount, canStartAnalysis, maxItems }: {
  analyses: LibraryAnalysis[];
  totalCount: number;
  canStartAnalysis: boolean;
  maxItems: number;
}) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [source, setSource] = useState<SourceFilter>("all");

  const counts = useMemo(() => {
    const completed = analyses.filter((analysis) => statusGroup(analysis.status) === "completed").length;
    const processing = analyses.filter((analysis) => statusGroup(analysis.status) === "processing").length;
    const failed = analyses.filter((analysis) => statusGroup(analysis.status) === "failed").length;
    return { completed, processing, failed };
  }, [analyses]);

  const filtered = useMemo(() => {
    const cleanQuery = query.trim().toLowerCase();
    return analyses.filter((analysis) => {
      const matchesQuery = !cleanQuery || analysis.searchText.includes(cleanQuery);
      const matchesStatus = status === "all" || statusGroup(analysis.status) === status;
      const matchesSource = source === "all" || analysis.sourceKind === source;
      return matchesQuery && matchesStatus && matchesSource;
    });
  }, [analyses, query, source, status]);

  const activeFilters = query.trim() || status !== "all" || source !== "all";

  return (
    <>
      <section className={styles.metrics} aria-label="Analysis history overview">
        <article><span>Total analyses</span><strong>{totalCount}</strong><small>Saved in this workspace</small></article>
        <article><span>Completed</span><strong>{counts.completed}</strong><small>Ready to review</small></article>
        <article><span>Processing</span><strong>{counts.processing}</strong><small>Queued or running</small></article>
        <article><span>Needs attention</span><strong>{counts.failed}</strong><small>Failed or canceled</small></article>
      </section>

      <section className={styles.libraryPanel} aria-labelledby="history-title">
        <div className={styles.panelHeading}>
          <div>
            <h2 id="history-title">Analysis history</h2>
            <p>Search by matchup, team or competition, then open the analysis directly.</p>
          </div>
          {canStartAnalysis ? <Link className={styles.primaryButton} href="/analysis/new">New analysis</Link> : null}
        </div>

        <div className={styles.controls}>
          <label className={styles.searchField}>
            <span className="sr-only">Search analyses</span>
            <input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search matchup, team or competition" />
          </label>
          <label className={styles.selectField}>
            <span>Status</span>
            <select value={status} onChange={(event) => setStatus(event.target.value as StatusFilter)}>
              <option value="all">All statuses</option>
              <option value="completed">Completed</option>
              <option value="processing">Processing</option>
              <option value="failed">Needs attention</option>
            </select>
          </label>
          <label className={styles.selectField}>
            <span>Source</span>
            <select value={source} onChange={(event) => setSource(event.target.value as SourceFilter)}>
              <option value="all">All sources</option>
              <option value="upload">Uploaded video</option>
              <option value="youtube">YouTube</option>
              <option value="other">Other</option>
            </select>
          </label>
          {activeFilters ? (
            <button type="button" className={styles.clearButton} onClick={() => { setQuery(""); setStatus("all"); setSource("all"); }}>
              Clear filters
            </button>
          ) : null}
        </div>

        <div className={styles.resultSummary} aria-live="polite">
          <span>{filtered.length} {filtered.length === 1 ? "analysis" : "analyses"}</span>
          {totalCount > maxItems ? <span>Showing the newest {maxItems} of {totalCount} analyses.</span> : null}
        </div>

        {filtered.length ? (
          <div className={styles.analysisList}>
            {filtered.map((analysis) => {
              const retention = retentionCopy(analysis);
              const duration = formatMinutes(analysis.durationMinutes);
              const retryable = canStartAnalysis && (analysis.status === "failed" || analysis.status === "canceled");
              const actionLabel = retryable
                ? "Edit and retry"
                : analysis.status === "completed"
                  ? "View results"
                  : analysis.status === "queued" || analysis.status === "running"
                    ? "Track analysis"
                    : "Open analysis";
              const actionHref = retryable ? `/analysis/new?retry=${analysis.id}` : `/analysis/${analysis.id}`;

              return (
                <article className={styles.analysisCard} key={analysis.id}>
                  <div className={styles.cardMain}>
                    <div className={styles.cardTopline}>
                      <span className={statusClass(analysis.status)}>{statusLabel(analysis.status)}</span>
                      <span className={styles.sourceBadge}>{sourceLabel(analysis.sourceKind)}</span>
                    </div>
                    <h3>{analysis.title}</h3>
                    <div className={styles.meta}>
                      <span>{formatDate(analysis.gameDate || analysis.createdAt)}</span>
                      {analysis.competition ? <span>{analysis.competition}</span> : null}
                      {duration ? <span>{duration} processed</span> : null}
                    </div>
                    <div className={`${styles.retention} ${retention.className}`}>{retention.label}</div>
                  </div>
                  <div className={styles.cardAction}>
                    <Link href={actionHref}>{actionLabel}<span aria-hidden="true">→</span></Link>
                    {retryable ? <small>Previous run stays in history</small> : <small>Analysis {analysis.id.slice(0, 8)}</small>}
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <div className={styles.emptyState}>
            <strong>{analyses.length ? "No analyses match those filters." : "No analyses yet."}</strong>
            <p>{analyses.length ? "Try a different team, competition, status or source." : "Your completed and in-progress basketball analyses will appear here after your first submission."}</p>
            {analyses.length ? (
              <button type="button" className={styles.secondaryButton} onClick={() => { setQuery(""); setStatus("all"); setSource("all"); }}>
                Show all analyses
              </button>
            ) : canStartAnalysis ? (
              <Link className={styles.primaryButton} href="/analysis/new">Start first analysis</Link>
            ) : null}
          </div>
        )}
      </section>
    </>
  );
}
