"use client";

import { useEffect, useMemo, useState } from "react";
import defaultMatchConfig from "@/data/match_config.json";

type WorkerHealth = {
  status?: string;
  error?: string;
  devices?: { mps?: boolean; cuda?: boolean; cpu?: boolean };
};

type Summary = {
  run_id?: string;
  elapsed_sec?: number;
  total_frames?: number;
  gameplay_frames?: number;
  players?: {
    raw_per_gameplay_frame?: number;
    parsed_per_gameplay_frame?: number;
    final_per_gameplay_frame?: number;
    team_classification_rate?: number;
    jersey_identification_rate?: number;
    team_counts?: Record<string, number>;
  };
  ball?: {
    raw_candidate_frame_rate?: number;
    selected_detection_frame_rate?: number;
    predicted_hold_frame_rate?: number;
    missing_frame_rate?: number;
    selected_conf_mean?: number | null;
    selected_conf_median?: number | null;
  };
  extra?: { status?: string; error?: string };
  [key: string]: unknown;
};

type JobState = {
  job_id?: string;
  status?: "queued" | "running" | "completed" | "failed" | string;
  error?: string;
  summary?: Summary | null;
};

const DEFAULT_VIDEO = "https://www.youtube.com/watch?v=OwiQgAMqyUs";

function pct(value: number | undefined) {
  return `${Number(value ?? 0).toFixed(1)}%`;
}

function metric(value: number | undefined, digits = 2) {
  return Number(value ?? 0).toFixed(digits);
}

export default function Home() {
  const [health, setHealth] = useState<WorkerHealth>({});
  const [videoSource, setVideoSource] = useState(DEFAULT_VIDEO);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(60);
  const [model, setModel] = useState("weights/Model_v0.1.pt");
  const [confidence, setConfidence] = useState(0.1);
  const [imgsz, setImgsz] = useState(640);
  const [device, setDevice] = useState<"mps" | "cpu" | "cuda">("mps");
  const [tracker, setTracker] = useState<"botsort.yaml" | "bytetrack.yaml">("botsort.yaml");
  const [matchConfigText, setMatchConfigText] = useState(
    JSON.stringify(defaultMatchConfig, null, 2),
  );
  const [job, setJob] = useState<JobState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");

  async function refreshHealth() {
    try {
      const response = await fetch("/api/health", { cache: "no-store" });
      setHealth(await response.json());
    } catch (error) {
      setHealth({ status: "error", error: error instanceof Error ? error.message : String(error) });
    }
  }

  useEffect(() => {
    void refreshHealth();
  }, []);

  useEffect(() => {
    if (!job?.job_id || ["completed", "failed"].includes(job.status ?? "")) return;

    const jobId = job.job_id;
    let cancelled = false;

    async function poll() {
      try {
        const [jobResponse, logResponse] = await Promise.all([
          fetch(`/api/jobs/${jobId}`, { cache: "no-store" }),
          fetch(`/api/jobs/${jobId}/log?tail=120`, { cache: "no-store" }),
        ]);

        if (!cancelled) {
          const nextJob = await jobResponse.json();
          setJob(nextJob);
          if (logResponse.ok) {
            const logData = await logResponse.json();
            setLogs(Array.isArray(logData.lines) ? logData.lines : []);
          }
        }
      } catch (error) {
        if (!cancelled) setFormError(error instanceof Error ? error.message : String(error));
      }
    }

    void poll();
    const timer = window.setInterval(() => void poll(), 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  const parsedMatchConfig = useMemo(() => {
    try {
      return { value: JSON.parse(matchConfigText), error: "" };
    } catch (error) {
      return { value: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [matchConfigText]);

  async function submitJob() {
    setFormError("");
    setLogs([]);

    if (!parsedMatchConfig.value) {
      setFormError(`Invalid match_config.json: ${parsedMatchConfig.error}`);
      return;
    }
    if (endTime <= startTime) {
      setFormError("End time must be greater than start time.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_source: videoSource,
          start_time_sec: startTime,
          end_time_sec: endTime,
          model,
          detector_conf: confidence,
          imgsz,
          device,
          tracker,
          match_config: parsedMatchConfig.value,
        }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? data.error ?? `HTTP ${response.status}`);
      setJob(data);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  const summary = job?.summary ?? null;
  const players = summary?.players;
  const ball = summary?.ball;
  const processingFps = summary?.total_frames && summary?.elapsed_sec
    ? summary.total_frames / summary.elapsed_sec
    : 0;

  return (
    <main>
      <header>
        <div>
          <h1>Basketball AI Analytics</h1>
          <p>Vercel beta control plane. Results and logs only; inference runs on the Python worker.</p>
        </div>
        <button
          className={`badge ${health.status === "ok" ? "ok" : health.status ? "error" : ""}`}
          onClick={() => void refreshHealth()}
          type="button"
        >
          Worker: {health.status === "ok" ? "online" : "offline"}
          {health.devices?.mps ? " · MPS" : health.devices?.cuda ? " · CUDA" : health.status === "ok" ? " · CPU" : ""}
        </button>
      </header>

      <div className="grid">
        <section className="panel span-8">
          <h2>Analysis run</h2>
          <div className="form-grid">
            <div className="field full">
              <label>YouTube URL / worker-local video path</label>
              <input value={videoSource} onChange={(e) => setVideoSource(e.target.value)} />
            </div>
            <div className="field">
              <label>Start (seconds)</label>
              <input type="number" min="0" step="1" value={startTime} onChange={(e) => setStartTime(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>End (seconds)</label>
              <input type="number" min="1" step="1" value={endTime} onChange={(e) => setEndTime(Number(e.target.value))} />
            </div>
            <div className="field full">
              <label>Model path on worker</label>
              <input value={model} onChange={(e) => setModel(e.target.value)} />
            </div>
            <div className="field">
              <label>Detector confidence</label>
              <input type="number" min="0.01" max="0.95" step="0.01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} />
            </div>
            <div className="field">
              <label>Image size</label>
              <select value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))}>
                {[640, 768, 960, 1280].map((size) => <option key={size} value={size}>{size}</option>)}
              </select>
            </div>
            <div className="field">
              <label>Device</label>
              <select value={device} onChange={(e) => setDevice(e.target.value as "mps" | "cpu" | "cuda")}>
                <option value="mps">MPS</option>
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA</option>
              </select>
            </div>
            <div className="field">
              <label>Tracker</label>
              <select value={tracker} onChange={(e) => setTracker(e.target.value as "botsort.yaml" | "bytetrack.yaml")}>
                <option value="botsort.yaml">BoT-SORT</option>
                <option value="bytetrack.yaml">ByteTrack</option>
              </select>
            </div>
          </div>

          <div className="actions">
            <button
              className="primary"
              type="button"
              onClick={() => void submitJob()}
              disabled={submitting || health.status !== "ok" || ["queued", "running"].includes(job?.status ?? "")}
            >
              {submitting ? "Submitting…" : ["queued", "running"].includes(job?.status ?? "") ? "Analysis running…" : "Run analysis"}
            </button>
            <span className="muted">Recommended comparison: keep everything fixed and change one parameter per run.</span>
          </div>
          {formError ? <div className="error-box" style={{ marginTop: 14 }}>{formError}</div> : null}
        </section>

        <section className="panel span-4">
          <h2>match_config.json</h2>
          <p className="muted">This configuration is sent with the job and snapshotted by the worker for player/team distinction.</p>
          <textarea value={matchConfigText} onChange={(e) => setMatchConfigText(e.target.value)} spellCheck={false} />
          {parsedMatchConfig.error ? <div className="error-box">{parsedMatchConfig.error}</div> : <div className="muted">JSON valid</div>}
        </section>

        <section className="panel span-12">
          <div className="status-line">
            <h2 style={{ margin: 0 }}>Results</h2>
            {job?.status ? <span className={`status ${job.status}`}>{job.status}</span> : <span className="muted">No remote run submitted yet.</span>}
            {job?.job_id ? <span className="muted">Job {job.job_id}</span> : null}
          </div>

          {job?.error ? <div className="error-box" style={{ marginTop: 14 }}>{job.error}</div> : null}

          {summary ? (
            <>
              <div className="metrics" style={{ marginTop: 18 }}>
                <div className="metric"><div className="label">Gameplay frames</div><div className="value">{summary.gameplay_frames ?? 0}</div></div>
                <div className="metric"><div className="label">Processing FPS</div><div className="value">{metric(processingFps)}</div></div>
                <div className="metric"><div className="label">Raw players / frame</div><div className="value">{metric(players?.raw_per_gameplay_frame, 3)}</div></div>
                <div className="metric"><div className="label">Final players / frame</div><div className="value">{metric(players?.final_per_gameplay_frame, 3)}</div></div>
                <div className="metric"><div className="label">Team classified</div><div className="value">{pct(players?.team_classification_rate)}</div></div>
                <div className="metric"><div className="label">Jersey identified</div><div className="value">{pct(players?.jersey_identification_rate)}</div></div>
                <div className="metric"><div className="label">Raw ball candidate</div><div className="value">{pct(ball?.raw_candidate_frame_rate)}</div></div>
                <div className="metric"><div className="label">Selected ball</div><div className="value">{pct(ball?.selected_detection_frame_rate)}</div></div>
                <div className="metric"><div className="label">Ball held/predicted</div><div className="value">{pct(ball?.predicted_hold_frame_rate)}</div></div>
                <div className="metric"><div className="label">Ball missing</div><div className="value">{pct(ball?.missing_frame_rate)}</div></div>
                <div className="metric"><div className="label">Ball mean confidence</div><div className="value">{metric(ball?.selected_conf_mean ?? undefined, 3)}</div></div>
                <div className="metric"><div className="label">Elapsed</div><div className="value">{metric(summary.elapsed_sec, 1)}s</div></div>
              </div>

              {players?.team_counts ? (
                <div style={{ marginTop: 18 }}>
                  <div className="muted" style={{ marginBottom: 8 }}>Team detection counts</div>
                  <pre>{JSON.stringify(players.team_counts, null, 2)}</pre>
                </div>
              ) : null}

              <details style={{ marginTop: 18 }}>
                <summary className="muted">Raw summary JSON</summary>
                <div style={{ marginTop: 10 }}><pre>{JSON.stringify(summary, null, 2)}</pre></div>
              </details>
            </>
          ) : null}
        </section>

        <section className="panel span-12">
          <h2>Worker log</h2>
          <pre>{logs.length ? logs.join("\n") : "Logs will appear here while the worker processes the job."}</pre>
        </section>
      </div>
    </main>
  );
}
