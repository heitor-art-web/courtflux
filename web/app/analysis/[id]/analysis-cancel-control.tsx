"use client";

import { useEffect, useState } from "react";
import styles from "./analysis-cancel-control.module.css";

type AnalysisState = {
  run?: { status?: string };
  worker?: { progress?: { phase?: string | null } | null } | null;
  error?: string;
};

type CancelResponse = {
  status?: string;
  error?: string;
};

function isActive(status: string | null) {
  return status === "queued" || status === "running" || status === "cancel_requested";
}

export function AnalysisCancelControl({ runId }: { runId: string }) {
  const [status, setStatus] = useState<string | null>(null);
  const [phase, setPhase] = useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const response = await fetch(`/api/analyses/${encodeURIComponent(runId)}`, { cache: "no-store" });
        const payload = (await response.json()) as AnalysisState;
        if (!response.ok) throw new Error(payload.error || `Could not load analysis (HTTP ${response.status})`);
        if (cancelled) return;

        const nextStatus = payload.run?.status ?? null;
        setStatus((current) => current === "cancel_requested" && isActive(nextStatus) ? current : nextStatus);
        setPhase(payload.worker?.progress?.phase ?? null);

        if (isActive(nextStatus)) {
          timer = window.setTimeout(() => void poll(), 3000);
        }
      } catch {
        if (!cancelled) timer = window.setTimeout(() => void poll(), 6000);
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [runId]);

  async function stopAnalysis() {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`/api/analyses/${encodeURIComponent(runId)}/cancel`, {
        method: "POST",
      });
      const payload = (await response.json()) as CancelResponse;
      if (!response.ok && response.status !== 202) {
        throw new Error(payload.error || `Could not stop analysis (HTTP ${response.status})`);
      }

      if (payload.status === "canceled") {
        setStatus("canceled");
        setConfirmOpen(false);
        window.location.reload();
        return;
      }

      setStatus("cancel_requested");
      setConfirmOpen(false);
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : String(cancelError));
    } finally {
      setBusy(false);
    }
  }

  if (!isActive(status)) return null;

  const stopping = status === "cancel_requested" || phase === "canceling";
  const queued = status === "queued";

  return (
    <>
      <div className={styles.floatingControl} aria-live="polite">
        {stopping ? (
          <div className={styles.stoppingPill}>
            <span className={styles.spinner} aria-hidden="true" />
            Stopping analysis…
          </div>
        ) : (
          <button className={styles.stopButton} type="button" onClick={() => { setError(""); setConfirmOpen(true); }}>
            {queued ? "Cancel analysis" : "Stop analysis"}
          </button>
        )}
        {error ? <div className={styles.inlineError} role="alert">{error}</div> : null}
      </div>

      {confirmOpen ? (
        <div className={styles.backdrop} role="presentation" onMouseDown={() => !busy && setConfirmOpen(false)}>
          <section
            className={styles.dialog}
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="cancel-analysis-title"
            aria-describedby="cancel-analysis-description"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <div className={styles.eyebrow}>{queued ? "Cancel queued analysis" : "Stop processing"}</div>
            <h2 id="cancel-analysis-title">{queued ? "Cancel this analysis?" : "Stop this analysis?"}</h2>
            <p id="cancel-analysis-description">
              {queued
                ? "CourtFlux will remove this run from the worker queue. No analysis minutes will be recorded and the canceled run will stay in your history."
                : "CourtFlux will stop the active worker process. No analysis minutes will be recorded and you can edit the setup and try again from the saved run."}
            </p>
            <div className={styles.dialogActions}>
              <button className={styles.keepButton} type="button" disabled={busy} onClick={() => setConfirmOpen(false)}>
                Keep running
              </button>
              <button className={styles.confirmButton} type="button" disabled={busy} onClick={() => void stopAnalysis()}>
                {busy ? "Stopping…" : queued ? "Yes, cancel analysis" : "Yes, stop analysis"}
              </button>
            </div>
            {error ? <div className={styles.dialogError} role="alert">{error}</div> : null}
          </section>
        </div>
      ) : null}
    </>
  );
}
