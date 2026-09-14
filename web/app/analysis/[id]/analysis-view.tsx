"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import styles from "./page.module.css";

type BallSummary = {
  raw_candidate_frame_rate?: number;
  selected_detection_frame_rate?: number;
  predicted_hold_frame_rate?: number;
  missing_frame_rate?: number;
  selected_conf_mean?: number | null;
  selected_conf_median?: number | null;
};

type PlayerSummary = {
  raw_per_gameplay_frame?: number;
  parsed_per_gameplay_frame?: number;
  final_per_gameplay_frame?: number;
  team_classification_rate?: number;
  jersey_identification_rate?: number;
  jersey_attempts?: number;
  jersey_successes?: number;
  team_counts?: Record<string, number>;
};

type PerformanceSummary = {
  analysis_fps?: number;
  realtime_factor?: number;
  stages?: Record<string, { calls?: number; total_sec?: number; avg_ms?: number; wall_pct?: number }>;
};

type Summary = {
  elapsed_sec?: number;
  total_frames?: number;
  gameplay_frames?: number;
  detection_only?: boolean;
  jersey_enabled?: boolean;
  performance?: PerformanceSummary;
  detector_timing?: Record<string, { calls?: number; total_sec?: number; avg_ms?: number }>;
  players?: PlayerSummary;
  ball?: BallSummary;
  extra?: { status?: string; error?: string; mode?: string };
  [key: string]: unknown;
};

type WorkerProgress = {
  phase: string | null;
  percent: number | null;
  processed_seconds: number | null;
  total_seconds: number | null;
  frames_processed: number | null;
  elapsed_sec: number | null;
  eta_sec: number | null;
  estimated_completion_at: string | null;
  last_progress_at: string | null;
};

type AnalysisData = {
  run: {
    id: string;
    status: string;
    start_time_sec: number | null;
    end_time_sec: number | null;
    parameters: Record<string, unknown> | null;
    summary: Summary | null;
    started_at: string | null;
    completed_at: string | null;
    created_at: string;
    updated_at: string;
  };
  job: {
    id: string;
    status: string;
    error: string | null;
    log_tail: string | null;
    queued_at: string | null;
    started_at: string | null;
    completed_at: string | null;
    created_at: string;
    updated_at: string;
  } | null;
  worker: {
    status: string | null;
    started_at: string | null;
    completed_at: string | null;
    progress: WorkerProgress | null;
    artifacts: {
      preview: boolean;
      analysis_video: boolean;
    } | null;
  } | null;
  game: {
    id: string;
    title: string | null;
    competition: string | null;
    game_date: string | null;
    source_type: string | null;
    metadata: Record<string, unknown> | null;
  } | null;
  source: {
    kind: "upload" | "youtube" | "other" | "unknown";
    playable_url: string | null;
    retention: {
      expires_at: string | null;
      deleted_at: string | null;
    } | null;
  };
  error?: string;
};

type Tab = "overview" | "ball" | "players" | "processing";

function clampPercent(value: number) {
  return Math.max(0, Math.min(100, value));
}

function asPercent(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  const percent = value >= 0 && value <= 1 ? value * 100 : value;
  return clampPercent(percent);
}

function formatPercent(value: number | null | undefined) {
  return `${asPercent(value).toFixed(1)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(digits).replace(/\.00$/, "").replace(/(\.\d)0$/, "$1");
}

function formatDateTime(value: string | null | undefined) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatExpectedTime(value: string | null | undefined) {
  if (!value) return "Calculating…";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Calculating…";
  return new Intl.DateTimeFormat("en", { hour: "2-digit", minute: "2-digit" }).format(date);
}

function formatClock(seconds: number | null | undefined) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${minutes}:${String(secs).padStart(2, "0")}`;
}

function formatCompactDuration(seconds: number | null | undefined) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "Calculating…";
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return secs >= 30 ? `${minutes + 1} min` : `${minutes}m ${secs}s`;
}

function formatDuration(start: number | null, end: number | null) {
  if (typeof start !== "number" || typeof end !== "number" || end <= start) return "—";
  const minutes = (end - start) / 60;
  return minutes < 10 ? `${minutes.toFixed(1).replace(/\.0$/, "")} min` : `${Math.round(minutes)} min`;
}

function statusClass(status: string) {
  if (status === "completed") return `${styles.statusPill} ${styles.statusCompleted}`;
  if (status === "failed" || status === "canceled") return `${styles.statusPill} ${styles.statusFailed}`;
  if (status === "running") return `${styles.statusPill} ${styles.statusRunning}`;
  return `${styles.statusPill} ${styles.statusQueued}`;
}

function statusCopy(status: string) {
  if (status === "completed") return "Tracking evidence is ready to review.";
  if (status === "running") return "CourtFlux is processing the requested video window now.";
  if (status === "failed") return "The worker could not complete this analysis.";
  if (status === "canceled") return "This analysis was canceled.";
  return "The analysis is queued and waiting for worker capacity.";
}

function flattenNumeric(value: unknown, prefix = "", rows: Array<[string, number]> = []) {
  if (typeof value === "number" && Number.isFinite(value) && prefix) {
    rows.push([prefix, value]);
    return rows;
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      flattenNumeric(child, prefix ? `${prefix}.${key}` : key, rows);
    }
  }
  return rows;
}

function csvCell(value: string | number) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function AnalysisView({ runId }: { runId: string }) {
  const [data, setData] = useState<AnalysisData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("overview");
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const response = await fetch(`/api/analyses/${encodeURIComponent(runId)}`, { cache: "no-store" });
        const payload = (await response.json()) as AnalysisData;
        if (!response.ok) throw new Error(payload.error || `Could not load analysis (HTTP ${response.status})`);
        if (cancelled) return;

        setData(payload);
        setRefreshTick((value) => value + 1);
        setError("");
        setLoading(false);

        if (["queued", "running"].includes(payload.run.status)) {
          timer = window.setTimeout(() => void poll(), 3000);
        }
      } catch (loadError) {
        if (cancelled) return;
        setError(loadError instanceof Error ? loadError.message : String(loadError));
        setLoading(false);
        timer = window.setTimeout(() => void poll(), 6000);
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [runId]);

  const summary = data?.run.summary ?? null;
  const ball = summary?.ball;
  const players = summary?.players;
  const performance = summary?.performance;
  const directBall = asPercent(ball?.selected_detection_frame_rate);
  const predictedBall = asPercent(ball?.predicted_hold_frame_rate);
  const effectiveBall = clampPercent(directBall + predictedBall);
  const missingBall = ball?.missing_frame_rate == null
    ? clampPercent(100 - effectiveBall)
    : asPercent(ball.missing_frame_rate);
  const confidenceMean = asPercent(ball?.selected_conf_mean ?? undefined);
  const teamClassified = asPercent(players?.team_classification_rate);
  const jerseyIdentified = asPercent(players?.jersey_identification_rate);
  const title = data?.game?.title || "Basketball analysis";
  const status = data?.run.status || "loading";
  const isActive = status === "queued" || status === "running";
  const isComplete = status === "completed";
  const isFailed = status === "failed" || status === "canceled";
  const sourceDeleted = Boolean(data?.source.retention?.deleted_at);
  const workerProgress = data?.worker?.progress ?? null;
  const progressPercent = isComplete
    ? 100
    : typeof workerProgress?.percent === "number"
      ? clampPercent(workerProgress.percent)
      : 0;
  const previewAvailable = Boolean(data?.worker?.artifacts?.preview);
  const analysedVideoAvailable = Boolean(data?.worker?.artifacts?.analysis_video);
  const lastProgressMs = workerProgress?.last_progress_at ? new Date(workerProgress.last_progress_at).getTime() : Number.NaN;
  const progressStalled = status === "running"
    && Number.isFinite(lastProgressMs)
    && Date.now() - lastProgressMs > 120_000
    && progressPercent < 99;
  const liveElapsed = status === "running" && data?.job?.started_at
    ? Math.max(0, (Date.now() - new Date(data.job.started_at).getTime()) / 1000)
    : workerProgress?.elapsed_sec;

  const sourceLabel = useMemo(() => {
    if (!data) return "—";
    if (data.source.kind === "upload") return sourceDeleted ? "Uploaded video · removed" : "Uploaded video";
    if (data.source.kind === "youtube") return "YouTube";
    if (data.source.kind === "other") return "Worker video source";
    return "Video source";
  }, [data, sourceDeleted]);

  function exportJson() {
    if (!data || !summary) return;
    downloadText(
      `courtflux-${runId.slice(0, 8)}-summary.json`,
      JSON.stringify(
        {
          analysis_id: data.run.id,
          game: data.game,
          status: data.run.status,
          start_time_sec: data.run.start_time_sec,
          end_time_sec: data.run.end_time_sec,
          summary,
        },
        null,
        2,
      ),
      "application/json;charset=utf-8",
    );
  }

  function exportCsv() {
    if (!summary) return;
    const rows = flattenNumeric(summary);
    const csv = ["metric,value", ...rows.map(([key, value]) => `${csvCell(key)},${csvCell(value)}`)].join("\n");
    downloadText(`courtflux-${runId.slice(0, 8)}-metrics.csv`, csv, "text/csv;charset=utf-8");
  }

  if (loading && !data) {
    return (
      <main className={styles.shell}>
        <div className={styles.loading} role="status" aria-live="polite">
          <div><strong>Loading CourtFlux analysis…</strong><span>Checking the current processing state.</span></div>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className={styles.shell}>
        <header className={styles.topbar}>
          <Link href="/" className={styles.brand} aria-label="CourtFlux home">
            <Image src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux" width={660} height={185} priority />
          </Link>
          <div className={styles.topActions}><Link href="/account">Dashboard</Link></div>
        </header>
        <div className={styles.errorBox} role="alert">{error || "Analysis not found."}</div>
      </main>
    );
  }

  const trackingNote = status === "running"
    ? workerProgress?.phase === "preparing"
      ? "Preparing source and analysis model"
      : workerProgress?.phase === "finalizing"
        ? "Tracking complete · finalizing analysed video"
        : typeof workerProgress?.percent === "number"
          ? `${Math.round(progressPercent)}% · ${formatClock(workerProgress.processed_seconds)} of ${formatClock(workerProgress.total_seconds)} analysed`
          : "Player and ball tracking in progress"
    : isComplete
      ? "Tracking pass completed"
      : isFailed
        ? "Processing stopped"
        : "Starts after queue";

  const steps = [
    { label: "Received", note: "Analysis request saved", state: "done" },
    {
      label: "Worker queue",
      note: status === "queued" ? "Waiting for capacity" : "Worker accepted the job",
      state: status === "queued" ? "active" : "done",
    },
    {
      label: "Track video",
      note: trackingNote,
      state: status === "running" ? "active" : isComplete ? "done" : "idle",
    },
    {
      label: "Results",
      note: isComplete ? "Evidence ready" : isFailed ? "No completed result" : "Available after processing",
      state: isComplete ? "done" : "idle",
    },
  ];

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Link href="/" className={styles.brand} aria-label="CourtFlux home">
          <Image src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux — Basketball Intelligence" width={660} height={185} priority />
        </Link>
        <div className={styles.topActions}>
          <Link href="/analysis/new">New analysis</Link>
          <Link href="/account">Dashboard</Link>
        </div>
      </header>

      <section className={styles.hero} aria-labelledby="analysis-title">
        <div>
          <div className={styles.eyebrow}>{isComplete ? "Analysis results" : "Processing analysis"}</div>
          <h1 id="analysis-title">{title}</h1>
          <p>
            {isComplete
              ? "Review the tracking evidence CourtFlux can currently support from this video. Unavailable basketball-event and tactical layers are kept clearly separate from measured beta output."
              : "You can leave this page and return later. The run is durable in your CourtFlux workspace and this page updates automatically while processing is active."}
          </p>
        </div>
        <aside className={styles.statusCard} aria-live="polite">
          <div className={styles.statusLine}>
            <span>Status</span>
            <span className={statusClass(status)}>{status}</span>
          </div>
          <strong>{statusCopy(status)}</strong>
          <small>{isActive ? "Status refreshes automatically every few seconds." : `Last updated ${formatDateTime(data.run.updated_at)}`}</small>
        </aside>
      </section>

      {error ? <div className={styles.infoBox} role="status">Latest refresh issue: {error}. The last known analysis state is still shown below.</div> : null}
      {isFailed ? (
        <div className={styles.errorBox} role="alert">
          <strong>Analysis did not complete.</strong>{" "}
          {data.job?.error || "The processing worker stopped before a complete tracking summary was produced."}
          {" "}Your allowance is only recorded when a run completes. You can adjust the source or analysis window and try again.
        </div>
      ) : null}

      <section className={styles.progressPanel} aria-labelledby="progress-title">
        <div className={styles.progressHeader}>
          <div>
            <h2 id="progress-title">Processing progress</h2>
            <p>Progress is based on the video timestamp actually processed by the worker. ETA appears once CourtFlux has enough tracking data to estimate the current pace.</p>
          </div>
        </div>
        <div className={styles.timeline}>
          {steps.map((step, index) => (
            <div className={`${styles.step} ${step.state === "active" ? styles.stepActive : ""} ${step.state === "done" ? styles.stepDone : ""}`} key={step.label}>
              <span className={styles.stepIndex}>{step.state === "done" ? "✓" : index + 1}</span>
              <strong>{step.label}</strong>
              <small>{step.note}</small>
            </div>
          ))}
        </div>

        {status === "running" ? (
          <div className={styles.progressDetail} aria-live="polite">
            {workerProgress?.phase === "preparing" || !workerProgress ? (
              <div className={styles.progressWaiting}>
                <strong>Preparing the analysis…</strong>
                <span>CourtFlux is preparing the source and model. A percentage and ETA will appear after frame tracking begins.</span>
              </div>
            ) : (
              <>
                <div className={styles.progressMeterHeader}>
                  <strong>{workerProgress.phase === "finalizing" ? "Finalizing analysed video" : "Track video"}</strong>
                  <span>{Math.round(progressPercent)}%</span>
                </div>
                <div
                  className={styles.progressMeter}
                  role="progressbar"
                  aria-label="Video analysis progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(progressPercent)}
                >
                  <div className={styles.progressMeterFill} style={{ width: `${progressPercent}%` }} />
                </div>
                <div className={styles.progressStats}>
                  <div><span>Video analysed</span><strong>{formatClock(workerProgress.processed_seconds)} / {formatClock(workerProgress.total_seconds)}</strong></div>
                  <div><span>Elapsed</span><strong>{formatCompactDuration(liveElapsed)}</strong></div>
                  <div><span>Estimated remaining</span><strong>{workerProgress.phase === "finalizing" ? "Finalizing…" : workerProgress.eta_sec == null ? "Calculating…" : `~${formatCompactDuration(workerProgress.eta_sec)}`}</strong></div>
                  <div><span>Expected ready</span><strong>{workerProgress.phase === "finalizing" ? "Soon" : formatExpectedTime(workerProgress.estimated_completion_at)}</strong></div>
                </div>
              </>
            )}
            {progressStalled ? (
              <div className={styles.progressWarning}>
                Processing has not reported new video progress for over two minutes. The run is still marked running, but it may be slower than expected.
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className={styles.metaGrid} aria-label="Analysis details">
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Analysis window</div>
          <div className={styles.metricValue}>{formatDuration(data.run.start_time_sec, data.run.end_time_sec)}</div>
          <div className={styles.metricNote}>{formatClock(data.run.start_time_sec)} → {formatClock(data.run.end_time_sec)}</div>
        </article>
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Video source</div>
          <div className={styles.metricValue}>{sourceLabel}</div>
          <div className={styles.metricNote}>Private workspace analysis source</div>
        </article>
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Requested</div>
          <div className={styles.metricValue}>{formatDateTime(data.run.created_at).split(",")[0]}</div>
          <div className={styles.metricNote}>{formatDateTime(data.run.created_at)}</div>
        </article>
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Analysis ID</div>
          <div className={styles.metricValue}>{data.run.id.slice(0, 8)}</div>
          <div className={styles.metricNote}>Stable workspace run reference</div>
        </article>
      </section>

      <div className={styles.contentGrid}>
        <div className={styles.stack}>
          <section className={styles.videoCard} aria-labelledby="analysed-video-title">
            <div className={styles.panelHeader}>
              <div>
                <h2 id="analysed-video-title">CourtFlux analysed video</h2>
                <p>{isComplete ? "Review the annotated tracking output produced by this run." : "Watch the latest annotated frame while CourtFlux processes the video. The final analysed video becomes playable when the run completes."}</p>
              </div>
            </div>
            <div className={styles.videoFrame}>
              {isComplete && analysedVideoAvailable ? (
                <video src={`/api/analyses/${encodeURIComponent(runId)}/preview-video`} controls preload="metadata">Your browser does not support video playback.</video>
              ) : previewAvailable ? (
                <img src={`/api/analyses/${encodeURIComponent(runId)}/preview?v=${refreshTick}`} alt="Latest CourtFlux annotated analysis preview" />
              ) : (
                <div className={styles.videoPlaceholder}>
                  <strong>{isComplete ? "Analysed video unavailable" : "Preparing analysed preview"}</strong>
                  <span>{isComplete ? "The worker has not exposed a final analysed video for this run." : "The first annotated frame will appear after CourtFlux starts tracking the video."}</span>
                </div>
              )}
            </div>
            {isActive && previewAvailable ? (
              <div className={styles.livePreviewMeta}><span className={styles.liveDot} aria-hidden="true" />Live annotated preview · refreshes with analysis progress</div>
            ) : null}
            {isComplete && !analysedVideoAvailable && previewAvailable ? (
              <div className={styles.infoBox}>The latest annotated frame is available, but the final analysed MP4 is not currently exposed by the worker.</div>
            ) : null}
          </section>

          <section className={styles.videoCard} aria-labelledby="video-title">
            <div className={styles.panelHeader}>
              <div>
                <h2 id="video-title">Source video</h2>
                <p>{data.source.kind === "youtube" ? "The original recording stays on YouTube and opens externally." : "Use the retained original footage alongside the CourtFlux analysis."}</p>
              </div>
            </div>

            {data.source.kind === "youtube" ? (
              <div className={styles.sourceExternalCard}>
                <div><strong>YouTube source</strong><span>Open the original recording in a new tab when you want to compare it with CourtFlux evidence.</span></div>
                {data.source.playable_url ? <a className={styles.externalLink} href={data.source.playable_url} target="_blank" rel="noreferrer">Open source video ↗</a> : null}
              </div>
            ) : (
              <div className={styles.videoFrame}>
                {data.source.kind === "upload" && data.source.playable_url ? (
                  <video src={data.source.playable_url} controls preload="metadata">Your browser does not support video playback.</video>
                ) : (
                  <div className={styles.videoPlaceholder}>
                    <strong>{sourceDeleted ? "Source video removed" : "Video preview unavailable"}</strong>
                    <span>{sourceDeleted ? "CourtFlux deleted this uploaded source after its 60-day retention period. The derived analysis results and statistics remain available." : "The worker source is not exposed as a browser-playable URL."}</span>
                  </div>
                )}
              </div>
            )}

            {data.source.kind === "upload" && data.source.retention ? (
              <div className={styles.infoBox}>
                <strong>Source-video retention.</strong>{" "}
                {sourceDeleted
                  ? `The source was deleted on ${formatDateTime(data.source.retention.deleted_at)}. Analysis results are retained.`
                  : `This uploaded source is scheduled for automatic deletion on ${formatDateTime(data.source.retention.expires_at)}. Analysis results and statistics will remain.`}
                {" "}<Link href="/video-data">Video & data policy</Link>
              </div>
            ) : null}
            <div className={styles.windowBar}>
              <strong>Processed window</strong>
              <div className={styles.windowTrack}><div className={styles.windowFill} /></div>
              <div className={styles.windowMeta}><span>{formatClock(data.run.start_time_sec)}</span><span>{formatDuration(data.run.start_time_sec, data.run.end_time_sec)}</span><span>{formatClock(data.run.end_time_sec)}</span></div>
            </div>
          </section>

          {isComplete && summary ? (
            <section className={styles.panel} aria-labelledby="evidence-title">
              <div className={styles.panelHeader}>
                <div><h2 id="evidence-title">Tracking evidence</h2><p>Diagnostics from this completed beta run, separated by evidence type.</p></div>
                <div className={styles.exportActions}>
                  <button className={styles.exportButton} type="button" onClick={exportCsv}>Export metrics CSV</button>
                  <button className={styles.exportButton} type="button" onClick={exportJson}>Export summary JSON</button>
                </div>
              </div>

              <div className={styles.tabs} role="tablist" aria-label="Result views">
                {(["overview", "ball", "players", "processing"] as Tab[]).map((item) => (
                  <button
                    key={item}
                    className={`${styles.tab} ${tab === item ? styles.tabActive : ""}`}
                    type="button"
                    role="tab"
                    aria-selected={tab === item}
                    onClick={() => setTab(item)}
                  >
                    {item === "overview" ? "Overview" : item === "ball" ? "Ball tracking" : item === "players" ? "Players" : "Processing"}
                  </button>
                ))}
              </div>

              {tab === "overview" ? (
                <>
                  <div className={styles.metaGrid}>
                    <article className={styles.metricCard}><div className={styles.metricLabel}>Gameplay frames</div><div className={styles.metricValue}>{formatNumber(summary.gameplay_frames, 0)}</div><div className={styles.metricNote}>Frames accepted as gameplay</div></article>
                    <article className={styles.metricCard}><div className={styles.metricLabel}>Direct ball coverage</div><div className={styles.metricValue}>{formatPercent(ball?.selected_detection_frame_rate)}</div><div className={styles.metricNote}>Frames with an associated ball detection</div></article>
                    <article className={styles.metricCard}><div className={styles.metricLabel}>Effective trajectory</div><div className={styles.metricValue}>{effectiveBall.toFixed(1)}%</div><div className={styles.metricNote}>Direct detection plus conservative predicted hold</div></article>
                    <article className={styles.metricCard}><div className={styles.metricLabel}>Team classified</div><div className={styles.metricValue}>{formatPercent(players?.team_classification_rate)}</div><div className={styles.metricNote}>Current team-label diagnostic</div></article>
                  </div>
                  <div className={styles.betaBox}>
                    <strong>Current beta interpretation</strong>
                    <ul className={styles.betaList}>
                      <li>Direct ball detections are observed evidence; predicted hold fills only conservative short gaps.</li>
                      <li>Team and jersey rates are diagnostic quality signals, not finished player-attribution statistics.</li>
                      <li>Possessions, passes, shots, rebounds and tactical actions are not presented as results until those layers are validated.</li>
                    </ul>
                  </div>
                </>
              ) : null}

              {tab === "ball" ? (
                <>
                  <div className={styles.coverageRow}><div className={styles.coverageLabel}><span>Raw ball candidate frames</span><strong>{formatPercent(ball?.raw_candidate_frame_rate)}</strong></div><div className={styles.barTrack}><div className={styles.barFill} style={{ width: `${asPercent(ball?.raw_candidate_frame_rate)}%` }} /></div></div>
                  <div className={styles.coverageRow}><div className={styles.coverageLabel}><span>Direct associated ball</span><strong>{directBall.toFixed(1)}%</strong></div><div className={styles.barTrack}><div className={styles.barFill} style={{ width: `${directBall}%` }} /></div></div>
                  <div className={styles.coverageRow}><div className={styles.coverageLabel}><span>Predicted short-gap hold</span><strong>{predictedBall.toFixed(1)}%</strong></div><div className={styles.barTrack}><div className={styles.barFill} style={{ width: `${predictedBall}%` }} /></div></div>
                  <div className={styles.coverageRow}><div className={styles.coverageLabel}><span>Effective trajectory coverage</span><strong>{effectiveBall.toFixed(1)}%</strong></div><div className={styles.barTrack}><div className={styles.barFill} style={{ width: `${effectiveBall}%` }} /></div></div>
                  <div className={styles.dataList}>
                    <div className={styles.dataRow}><span>Missing trajectory frames</span><strong>{missingBall.toFixed(1)}%</strong></div>
                    <div className={styles.dataRow}><span>Mean selected-ball confidence</span><strong>{ball?.selected_conf_mean == null ? "—" : `${confidenceMean.toFixed(1)}%`}</strong></div>
                    <div className={styles.dataRow}><span>Median selected-ball confidence</span><strong>{ball?.selected_conf_median == null ? "—" : formatPercent(ball.selected_conf_median)}</strong></div>
                  </div>
                </>
              ) : null}

              {tab === "players" ? (
                <div className={styles.dataList}>
                  <div className={styles.dataRow}><span>Raw players per gameplay frame</span><strong>{formatNumber(players?.raw_per_gameplay_frame, 3)}</strong></div>
                  <div className={styles.dataRow}><span>Parsed players per gameplay frame</span><strong>{formatNumber(players?.parsed_per_gameplay_frame, 3)}</strong></div>
                  <div className={styles.dataRow}><span>Final players per gameplay frame</span><strong>{formatNumber(players?.final_per_gameplay_frame, 3)}</strong></div>
                  <div className={styles.dataRow}><span>Team classification rate</span><strong>{teamClassified.toFixed(1)}%</strong></div>
                  <div className={styles.dataRow}><span>Jersey identification rate</span><strong>{jerseyIdentified.toFixed(1)}%</strong></div>
                  <div className={styles.dataRow}><span>Jersey attempts</span><strong>{formatNumber(players?.jersey_attempts, 0)}</strong></div>
                  <div className={styles.dataRow}><span>Jersey successes</span><strong>{formatNumber(players?.jersey_successes, 0)}</strong></div>
                </div>
              ) : null}

              {tab === "processing" ? (
                <div className={styles.dataList}>
                  <div className={styles.dataRow}><span>Total frames</span><strong>{formatNumber(summary.total_frames, 0)}</strong></div>
                  <div className={styles.dataRow}><span>Elapsed processing time</span><strong>{summary.elapsed_sec == null ? "—" : `${formatNumber(summary.elapsed_sec, 1)} sec`}</strong></div>
                  <div className={styles.dataRow}><span>Analysis FPS</span><strong>{formatNumber(performance?.analysis_fps, 2)}</strong></div>
                  <div className={styles.dataRow}><span>Real-time factor</span><strong>{performance?.realtime_factor == null ? "—" : `${formatNumber(performance.realtime_factor, 2)}x`}</strong></div>
                  <div className={styles.dataRow}><span>Mode</span><strong>{summary.detection_only ? "Detection-only beta" : summary.extra?.mode || "Full pipeline"}</strong></div>
                  <div className={styles.dataRow}><span>Jersey recognition</span><strong>{summary.jersey_enabled ? "Enabled" : "Disabled"}</strong></div>
                  <div className={styles.dataRow}><span>Started</span><strong>{formatDateTime(data.run.started_at || data.job?.started_at)}</strong></div>
                  <div className={styles.dataRow}><span>Completed</span><strong>{formatDateTime(data.run.completed_at || data.job?.completed_at)}</strong></div>
                </div>
              ) : null}
            </section>
          ) : (
            <section className={styles.panel}>
              <h2>Results will appear here</h2>
              <p className={styles.infoBox}>{isFailed ? "No complete tracking summary was produced for this run." : "CourtFlux will switch this page from processing to results automatically when the worker completes the run."}</p>
            </section>
          )}
        </div>

        <aside className={styles.stack} aria-label="Analysis context">
          <section className={styles.panel}>
            <div className={styles.eyebrow}>Game context</div>
            <div className={styles.dataList}>
              <div className={styles.dataRow}><span>Game</span><strong>{title}</strong></div>
              <div className={styles.dataRow}><span>Competition</span><strong>{data.game?.competition || "Not provided"}</strong></div>
              <div className={styles.dataRow}><span>Game date</span><strong>{data.game?.game_date ? formatDateTime(data.game.game_date).split(",")[0] : "Not provided"}</strong></div>
              <div className={styles.dataRow}><span>Source</span><strong>{sourceLabel}</strong></div>
              {data.source.kind === "upload" && data.source.retention ? <div className={styles.dataRow}><span>Source retention</span><strong>{sourceDeleted ? "Video removed · stats retained" : `Deletes ${formatDateTime(data.source.retention.expires_at)}`}</strong></div> : null}
            </div>
          </section>

          <section className={styles.panel}>
            <div className={styles.eyebrow}>Evidence boundary</div>
            <h2>What this page does not invent</h2>
            <ul className={styles.betaList}>
              <li>No possession or event attribution before the event primitives are validated.</li>
              <li>No player identity claims from anonymous tracks unless jersey/identity evidence supports them.</li>
              <li>No tactical recommendations presented as measured facts before the tactical layer exists.</li>
            </ul>
          </section>

          <section className={styles.panel}>
            <div className={styles.eyebrow}>Next action</div>
            <h2>{isComplete ? "Use the evidence, then analyse the next clip." : isFailed ? "Adjust and try again." : "You can leave this running."}</h2>
            <p className={styles.infoBox}>{isComplete ? "Export the current metrics or return to the dashboard. CourtFlux keeps the completed run attached to your workspace." : "This run remains saved in your workspace even if you close the page."}</p>
            <div className={styles.exportActions}>
              <Link className={styles.secondaryButton} href="/account">Dashboard</Link>
              <Link className={styles.secondaryButton} href="/analysis/new">New analysis</Link>
            </div>
          </section>
        </aside>
      </div>
    </main>
  );
}
