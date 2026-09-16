"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type LabDefaults = {
  video_source: string;
  model: string;
  detector_conf: number;
  inference_conf: number;
  ball_conf: number;
  imgsz: number;
  device: "mps" | "cpu" | "cuda";
  tracker: "botsort.yaml" | "bytetrack.yaml";
  enable_jersey: boolean;
  jersey_backend?: string;
  jersey_interval_frames: number;
  live_preview: boolean;
  preview_interval_frames: number;
  match_config?: Record<string, unknown>;
  match_config_source?: string;
};

type WorkerHealth = {
  status?: string;
  error?: string;
  supabase_persistence?: boolean;
  devices?: { mps?: boolean; cuda?: boolean; cpu?: boolean };
  lab_defaults?: LabDefaults;
};

type AttributionRow = {
  total?: number;
  actor_count?: number;
  actor_rate?: number;
  team_count?: number;
  team_rate?: number;
};

type P2Summary = {
  version?: string;
  frames_seen?: number;
  p2_processed_frames?: number;
  metric_eligible_frames?: number;
  engine_eligible_frames?: number;
  control_state?: {
    counts?: Record<string, number>;
    rates?: Record<string, number>;
  };
  possession?: {
    possessed_frames?: number;
    possessed_rate?: number;
    handler_memory_frames?: number;
    handler_memory_rate?: number;
    avg_possession_confidence?: number | null;
    changes?: number;
    acquisitions?: number;
  };
  events?: {
    total?: number;
    counts?: Record<string, number>;
    decisions?: Record<string, number>;
    mean_confidence_by_type?: Record<string, number>;
    actor_attribution_rate?: number;
    team_attribution_rate?: number;
    attribution_by_event_type?: Record<string, AttributionRow>;
  };
  shots?: {
    attempts?: number;
    outcomes?: Record<string, number>;
    unknown_outcome_rate?: number;
    rapid_unknown_retrigger_count?: number;
    minimum_inter_shot_gap_frames?: number | null;
  };
  transitions?: {
    passes?: number;
    passes_with_flight_evidence?: number;
    turnovers?: number;
    turnovers_with_flight_evidence?: number;
    direct_turnovers?: number;
    loose_ball_started?: number;
    loose_ball_recovered?: number;
    loose_ball_ended?: number;
    loose_ball_closed?: number;
    loose_ball_open_intervals?: number;
    loose_ball_closure_rate?: number;
    loose_ball_terminal_reasons?: Record<string, number>;
  };
  passes?: number;
  turnovers?: number;
  rebounds?: number;
  resets?: { count?: number; reasons?: Record<string, number> };
  p1_state_not_mutated_rate?: number;
};

type Summary = {
  run_id?: string;
  model?: string;
  inference_conf?: number;
  player_conf?: number;
  ball_conf?: number;
  imgsz?: number;
  elapsed_sec?: number;
  total_frames?: number;
  gameplay_frames?: number;
  jersey_enabled?: boolean;
  performance?: {
    analysis_fps?: number;
    realtime_factor?: number;
    stages?: Record<string, unknown>;
  };
  detector_timing?: Record<string, unknown>;
  players?: {
    parsed_per_gameplay_frame?: number;
    team_classification_rate?: number;
    jersey_identification_rate?: number;
    jersey_attempts?: number;
    team_counts?: Record<string, number>;
  };
  ball?: {
    raw_candidate_frame_rate?: number;
    selected_detection_frame_rate?: number;
    missing_frame_rate?: number;
    selected_conf_mean?: number | null;
    continuity?: Record<string, unknown>;
  };
  p2?: P2Summary;
  extra?: { status?: string; error?: string; mode?: string; identity_debug?: Record<string, unknown> };
  [key: string]: unknown;
};

type JobState = {
  job_id?: string;
  status?: "queued" | "running" | "completed" | "failed" | string;
  error?: string;
  summary?: Summary | null;
  artifacts?: { preview?: boolean; analysis_video?: boolean };
};

const FALLBACK: LabDefaults = {
  video_source: "https://www.youtube.com/watch?v=OwiQgAMqyUs",
  model: "weights/model_v1.1.0.pt",
  detector_conf: 0.4,
  inference_conf: 0.03,
  ball_conf: 0.075,
  imgsz: 960,
  device: "mps",
  tracker: "botsort.yaml",
  enable_jersey: true,
  jersey_backend: "best_temporal",
  jersey_interval_frames: 24,
  live_preview: true,
  preview_interval_frames: 15,
};

function pct(value: number | undefined) {
  return `${Number(value ?? 0).toFixed(1)}%`;
}

function metric(value: number | undefined | null, digits = 2) {
  return Number(value ?? 0).toFixed(digits);
}

function labelize(value: string) {
  return value.replaceAll("_", " ");
}

export default function LabPage() {
  const [health, setHealth] = useState<WorkerHealth>({});
  const defaultsApplied = useRef(false);
  const matchConfigApplied = useRef(false);
  const [videoSource, setVideoSource] = useState(FALLBACK.video_source);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(60);
  const [model, setModel] = useState(FALLBACK.model);
  const [confidence, setConfidence] = useState(FALLBACK.detector_conf);
  const [inferenceConf, setInferenceConf] = useState(FALLBACK.inference_conf);
  const [ballConf, setBallConf] = useState(FALLBACK.ball_conf);
  const [imgsz, setImgsz] = useState(FALLBACK.imgsz);
  const [device, setDevice] = useState<"mps" | "cpu" | "cuda">(FALLBACK.device);
  const [tracker, setTracker] = useState<"botsort.yaml" | "bytetrack.yaml">(FALLBACK.tracker);
  const [enableJersey, setEnableJersey] = useState(FALLBACK.enable_jersey);
  const [jerseyInterval, setJerseyInterval] = useState(FALLBACK.jersey_interval_frames);
  const [livePreview, setLivePreview] = useState(FALLBACK.live_preview);
  const [previewInterval, setPreviewInterval] = useState(FALLBACK.preview_interval_frames);
  const [previewNonce, setPreviewNonce] = useState(0);
  const [matchConfigText, setMatchConfigText] = useState("{}");
  const [job, setJob] = useState<JobState | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState("");

  function applyDefaults(d: LabDefaults) {
    setVideoSource(d.video_source);
    setModel(d.model);
    setConfidence(d.detector_conf);
    setInferenceConf(d.inference_conf);
    setBallConf(d.ball_conf);
    setImgsz(d.imgsz);
    setDevice(d.device);
    setTracker(d.tracker);
    setEnableJersey(d.enable_jersey);
    setJerseyInterval(d.jersey_interval_frames);
    setLivePreview(d.live_preview);
    setPreviewInterval(d.preview_interval_frames);
  }

  function applyCanonicalMatchConfig(d?: LabDefaults) {
    if (d?.match_config) setMatchConfigText(JSON.stringify(d.match_config, null, 2));
  }

  async function refreshHealth() {
    try {
      const response = await fetch("/api/lab/health", { cache: "no-store" });
      const data: WorkerHealth = await response.json();
      setHealth(data);
      if (data.status === "ok" && data.lab_defaults) {
        if (!defaultsApplied.current) {
          applyDefaults(data.lab_defaults);
          defaultsApplied.current = true;
        }
        if (!matchConfigApplied.current && data.lab_defaults.match_config) {
          applyCanonicalMatchConfig(data.lab_defaults);
          matchConfigApplied.current = true;
        }
      }
    } catch (error) {
      setHealth({ status: "error", error: error instanceof Error ? error.message : String(error) });
    }
  }

  useEffect(() => {
    void refreshHealth();
  }, []);

  useEffect(() => {
    if (!job?.job_id || ["completed", "failed", "canceled"].includes(job.status ?? "")) return;
    const jobId = job.job_id;
    let cancelled = false;
    async function poll() {
      try {
        const [jobResponse, logResponse] = await Promise.all([
          fetch(`/api/lab/jobs/${jobId}`, { cache: "no-store" }),
          fetch(`/api/lab/jobs/${jobId}/log?tail=120`, { cache: "no-store" }),
        ]);
        if (!cancelled) {
          setJob(await jobResponse.json());
          if (logResponse.ok) {
            const logData = await logResponse.json();
            setLogs(Array.isArray(logData.lines) ? logData.lines : []);
          }
          if (livePreview) setPreviewNonce(Date.now());
        }
      } catch (error) {
        if (!cancelled) setFormError(error instanceof Error ? error.message : String(error));
      }
    }
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status, livePreview]);

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
    if (inferenceConf > Math.min(confidence, ballConf)) {
      setFormError("YOLO inference confidence must be <= player and ball confidence.");
      return;
    }

    setSubmitting(true);
    try {
      const response = await fetch("/api/lab/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_source: videoSource,
          start_time_sec: startTime,
          end_time_sec: endTime,
          model,
          detector_conf: confidence,
          inference_conf: inferenceConf,
          ball_conf: ballConf,
          imgsz,
          device,
          tracker,
          enable_jersey: enableJersey,
          jersey_backend: health.lab_defaults?.jersey_backend ?? "best_temporal",
          jersey_interval_frames: jerseyInterval,
          live_preview: livePreview,
          preview_interval_frames: previewInterval,
          single_forward: true,
          smart_ball_mode: "off",
          diagnostics_batch_size: 100,
          match_config: parsedMatchConfig.value,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? data.error ?? `HTTP ${response.status}`);
      setJob(data);
      setPreviewNonce(Date.now());
    } catch (error) {
      setFormError(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  const summary = job?.summary ?? null;
  const players = summary?.players;
  const ball = summary?.ball;
  const performance = summary?.performance;
  const p2 = summary?.p2;
  const processingFps = performance?.analysis_fps ??
    (summary?.total_frames && summary?.elapsed_sec ? summary.total_frames / summary.elapsed_sec : 0);
  const analysisActive = ["queued", "running"].includes(job?.status ?? "");
  const workerLabel = health.status === "ok" ? "online" : health.status ? "offline" : "checking";
  const previewUrl = job?.job_id ? `/api/lab/jobs/${job.job_id}/preview?t=${previewNonce}` : "";
  const videoUrl = job?.job_id ? `/api/lab/jobs/${job.job_id}/preview-video` : "";
  const shotOutcomes = p2?.shots?.outcomes ?? {};
  const attributionRows = Object.entries(p2?.events?.attribution_by_event_type ?? {});

  return (
    <main>
      <header>
        <div>
          <h1>CourtFlux Beta Lab</h1>
          <p>Canonical P1 → P2 runtime · model v1.1 · Smart Ball OFF · Supabase persistence.</p>
        </div>
        <button className={`badge ${health.status === "ok" ? "ok" : health.status ? "error" : ""}`} onClick={() => void refreshHealth()} type="button">
          Worker: {workerLabel}{health.devices?.mps ? " · MPS" : health.devices?.cuda ? " · CUDA" : health.status === "ok" ? " · CPU" : ""}{health.supabase_persistence ? " · Supabase" : ""}
        </button>
      </header>

      <div className="grid">
        <section className="panel span-8">
          <div className="status-line"><h2 style={{ margin: 0 }}>Analysis run</h2><span className="muted">{model} · {imgsz}px</span></div>
          <div className="form-grid">
            <div className="field full"><label>Video source</label><input value={videoSource} onChange={(e) => setVideoSource(e.target.value)} /></div>
            <div className="field"><label>Start (seconds)</label><input type="number" min="0" value={startTime} onChange={(e) => setStartTime(Number(e.target.value))} /></div>
            <div className="field"><label>End (seconds)</label><input type="number" min="1" value={endTime} onChange={(e) => setEndTime(Number(e.target.value))} /></div>
            <div className="field full"><label>Model path</label><input value={model} onChange={(e) => setModel(e.target.value)} /></div>
            <div className="field"><label>Player confidence</label><input type="number" min="0.01" max="0.95" step="0.01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></div>
            <div className="field"><label>Inference confidence</label><input type="number" min="0.001" max="0.95" step="0.01" value={inferenceConf} onChange={(e) => setInferenceConf(Number(e.target.value))} /></div>
            <div className="field"><label>Ball confidence</label><input type="number" min="0.001" max="0.95" step="0.01" value={ballConf} onChange={(e) => setBallConf(Number(e.target.value))} /></div>
            <div className="field"><label>Image size</label><select value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))}>{[640, 768, 960, 1280].map((size) => <option key={size}>{size}</option>)}</select></div>
            <div className="field"><label>Device</label><select value={device} onChange={(e) => setDevice(e.target.value as "mps" | "cpu" | "cuda")}><option value="mps">MPS</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></div>
            <div className="field"><label>Tracker</label><select value={tracker} onChange={(e) => setTracker(e.target.value as "botsort.yaml" | "bytetrack.yaml")}><option value="botsort.yaml">BoT-SORT</option><option value="bytetrack.yaml">ByteTrack</option></select></div>
            <div className="field"><label><input type="checkbox" checked={enableJersey} onChange={(e) => setEnableJersey(e.target.checked)} /> Jersey recognition</label><span className="muted">Optional identity evidence; track IDs remain operational identity.</span></div>
            <div className="field"><label>Jersey interval</label><input type="number" min="1" max="300" value={jerseyInterval} disabled={!enableJersey} onChange={(e) => setJerseyInterval(Number(e.target.value))} /></div>
            <div className="field"><span>Persistence</span><strong>Supabase only</strong></div>
            <div className="field"><label><input type="checkbox" checked={livePreview} onChange={(e) => setLivePreview(e.target.checked)} /> Live preview</label></div>
            <div className="field"><label>Preview interval</label><input type="number" min="1" max="300" value={previewInterval} disabled={!livePreview} onChange={(e) => setPreviewInterval(Number(e.target.value))} /></div>
          </div>
          <div className="actions">
            <button className="primary" type="button" onClick={() => void submitJob()} disabled={submitting || health.status !== "ok" || analysisActive}>{submitting ? "Submitting…" : analysisActive ? "Analysis running…" : "Run analysis"}</button>
            <button type="button" onClick={() => applyDefaults(health.lab_defaults ?? FALLBACK)} disabled={analysisActive}>Reset defaults</button>
          </div>
          {formError ? <div className="error-box" role="alert" style={{ marginTop: 14 }}>{formError}</div> : null}
        </section>

        <section className="panel span-4">
          <div className="status-line"><h2 style={{ margin: 0 }}>match_config.json</h2><button type="button" onClick={() => applyCanonicalMatchConfig(health.lab_defaults)}>Reload</button></div>
          <p className="muted">Canonical source: {health.lab_defaults?.match_config_source ?? "data/match_config.json"}</p>
          <textarea value={matchConfigText} onChange={(e) => setMatchConfigText(e.target.value)} spellCheck={false} />
          {parsedMatchConfig.error ? <div className="error-box">{parsedMatchConfig.error}</div> : <div className="muted">JSON valid</div>}
        </section>

        {job?.job_id && livePreview ? <section className="panel span-12">
          <div className="status-line"><h2 style={{ margin: 0 }}>Visual QA</h2><span className="muted">Diagnostic only.</span></div>
          {analysisActive || job.artifacts?.preview ? <img src={previewUrl} alt="CourtFlux diagnostic preview" style={{ width: "100%", maxHeight: 680, objectFit: "contain", marginTop: 16, background: "#080b10" }} /> : null}
          {job.status === "completed" && job.artifacts?.analysis_video ? <video controls src={videoUrl} style={{ width: "100%", maxHeight: 680, marginTop: 18, background: "#080b10" }} /> : null}
        </section> : null}

        <section className="panel span-12">
          <div className="status-line"><h2 style={{ margin: 0 }}>Results</h2><span>{job?.status ?? "No run"}</span></div>
          {job?.error ? <div className="error-box">{job.error}</div> : null}
          {summary ? <>
            <p className="muted">Run {summary.run_id ?? "—"} · mode {summary.extra?.mode ?? "canonical"}</p>
            <div className="metrics">
              <div className="metric"><div className="label">Gameplay frames</div><div className="value">{summary.gameplay_frames ?? 0}</div></div>
              <div className="metric"><div className="label">Analysis FPS</div><div className="value">{metric(processingFps)}</div></div>
              <div className="metric"><div className="label">Real-time factor</div><div className="value">{metric(performance?.realtime_factor)}x</div></div>
              <div className="metric"><div className="label">Players / frame</div><div className="value">{metric(players?.parsed_per_gameplay_frame, 3)}</div></div>
              <div className="metric"><div className="label">Team classified</div><div className="value">{pct(players?.team_classification_rate)}</div></div>
              <div className="metric"><div className="label">Jersey identified</div><div className="value">{pct(players?.jersey_identification_rate)}</div></div>
              <div className="metric"><div className="label">Ball candidate frames</div><div className="value">{pct(ball?.raw_candidate_frame_rate)}</div></div>
              <div className="metric"><div className="label">Ball selected</div><div className="value">{pct(ball?.selected_detection_frame_rate)}</div></div>
              <div className="metric"><div className="label">Ball missing</div><div className="value">{pct(ball?.missing_frame_rate)}</div></div>
            </div>

            {p2 ? <>
              <div className="status-line" style={{ marginTop: 24 }}>
                <h2 style={{ margin: 0 }}>P2 basketball understanding</h2>
                <span className="muted">{p2.version ?? "p2"} · Jersey-independent track identity</span>
              </div>
              <div className="metrics" style={{ marginTop: 14 }}>
                <div className="metric"><div className="label">Possessed</div><div className="value">{pct(p2.possession?.possessed_rate)}</div></div>
                <div className="metric"><div className="label">Handler memory</div><div className="value">{pct(p2.possession?.handler_memory_rate)}</div></div>
                <div className="metric"><div className="label">Control UNKNOWN</div><div className="value">{pct(p2.control_state?.rates?.unknown)}</div></div>
                <div className="metric"><div className="label">Possession confidence</div><div className="value">{metric(p2.possession?.avg_possession_confidence, 3)}</div></div>
                <div className="metric"><div className="label">P2 events</div><div className="value">{p2.events?.total ?? 0}</div></div>
                <div className="metric"><div className="label">Actor attribution</div><div className="value">{pct(p2.events?.actor_attribution_rate)}</div></div>
                <div className="metric"><div className="label">Team attribution</div><div className="value">{pct(p2.events?.team_attribution_rate)}</div></div>
                <div className="metric"><div className="label">P1 invariant</div><div className="value">{pct(p2.p1_state_not_mutated_rate)}</div></div>
                <div className="metric"><div className="label">Shots</div><div className="value">{p2.shots?.attempts ?? 0}</div></div>
                <div className="metric"><div className="label">Shot outcomes M/M/U</div><div className="value">{shotOutcomes.made ?? 0}/{shotOutcomes.missed ?? 0}/{shotOutcomes.unknown ?? 0}</div></div>
                <div className="metric"><div className="label">Unknown shot outcome</div><div className="value">{pct(p2.shots?.unknown_outcome_rate)}</div></div>
                <div className="metric"><div className="label">Rapid shot retriggers</div><div className="value">{p2.shots?.rapid_unknown_retrigger_count ?? 0}</div></div>
                <div className="metric"><div className="label">Passes</div><div className="value">{p2.transitions?.passes ?? p2.passes ?? 0}</div></div>
                <div className="metric"><div className="label">Passes with flight</div><div className="value">{p2.transitions?.passes_with_flight_evidence ?? 0}</div></div>
                <div className="metric"><div className="label">Turnovers</div><div className="value">{p2.transitions?.turnovers ?? p2.turnovers ?? 0}</div></div>
                <div className="metric"><div className="label">Direct turnovers</div><div className="value">{p2.transitions?.direct_turnovers ?? 0}</div></div>
                <div className="metric"><div className="label">Rebounds</div><div className="value">{p2.rebounds ?? 0}</div></div>
                <div className="metric"><div className="label">Loose closure</div><div className="value">{pct(p2.transitions?.loose_ball_closure_rate)}</div></div>
                <div className="metric"><div className="label">Loose open</div><div className="value">{p2.transitions?.loose_ball_open_intervals ?? 0}</div></div>
                <div className="metric"><div className="label">Temporal resets</div><div className="value">{p2.resets?.count ?? 0}</div></div>
              </div>

              <details style={{ marginTop: 18 }} open>
                <summary>Control-state distribution</summary>
                <div className="metrics" style={{ marginTop: 12 }}>
                  {Object.entries(p2.control_state?.rates ?? {}).map(([state, rate]) => <div className="metric" key={state}><div className="label">{labelize(state)}</div><div className="value">{pct(rate)}</div></div>)}
                </div>
              </details>

              <details style={{ marginTop: 18 }} open>
                <summary>Attribution by event type</summary>
                {attributionRows.length ? <div style={{ overflowX: "auto", marginTop: 12 }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
                    <thead><tr><th>Event</th><th>Total</th><th>Actor</th><th>Team</th></tr></thead>
                    <tbody>{attributionRows.map(([eventType, row]) => <tr key={eventType}><td>{labelize(eventType)}</td><td>{row.total ?? 0}</td><td>{pct(row.actor_rate)}</td><td>{pct(row.team_rate)}</td></tr>)}</tbody>
                  </table>
                </div> : <p className="muted">No P2 events emitted in this run.</p>}
              </details>

              <details style={{ marginTop: 18 }}>
                <summary>Loose lifecycle and reset diagnostics</summary>
                <pre>{JSON.stringify({
                  loose: p2.transitions,
                  resets: p2.resets,
                  event_counts: p2.events?.counts,
                }, null, 2)}</pre>
              </details>
            </> : <p className="muted" style={{ marginTop: 18 }}>P2 metrics are unavailable for this older run.</p>}

            {summary.extra?.identity_debug ? <details style={{ marginTop: 18 }}><summary>Identity debug</summary><pre>{JSON.stringify(summary.extra.identity_debug, null, 2)}</pre></details> : null}
            <details style={{ marginTop: 18 }}><summary>Raw summary JSON</summary><pre>{JSON.stringify(summary, null, 2)}</pre></details>
          </> : null}
        </section>

        <section className="panel span-12"><h2>Worker log</h2><pre>{logs.length ? logs.join("\n") : "Logs will appear here while the worker processes the job."}</pre></section>
      </div>
    </main>
  );
}