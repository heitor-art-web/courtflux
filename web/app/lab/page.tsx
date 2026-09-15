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
  detection_only: boolean;
  enable_jersey: boolean;
  jersey_backend?: "best_temporal" | "auto" | "uncertainty_vit_b" | "legacy_digit_heads";
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
  detection_only?: boolean;
  jersey_enabled?: boolean;
  performance?: {
    analysis_fps?: number;
    realtime_factor?: number;
    stages?: Record<string, { calls?: number; total_sec?: number; avg_ms?: number; wall_pct?: number }>;
  };
  detector_timing?: Record<string, { calls?: number; total_sec?: number; avg_ms?: number }>;
  players?: {
    raw_per_gameplay_frame?: number;
    parsed_per_gameplay_frame?: number;
    final_per_gameplay_frame?: number;
    team_classification_rate?: number;
    jersey_identification_rate?: number;
    jersey_attempts?: number;
    jersey_successes?: number;
    team_counts?: Record<string, number>;
  };
  ball?: {
    raw_candidate_frame_rate?: number;
    selected_detection_frame_rate?: number;
    predicted_hold_frame_rate?: number;
    missing_frame_rate?: number;
    selected_conf_mean?: number | null;
    selected_conf_median?: number | null;
    continuity?: Record<string, unknown>;
  };
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
  detector_conf: 0.40,
  inference_conf: 0.03,
  ball_conf: 0.075,
  imgsz: 960,
  device: "mps",
  tracker: "botsort.yaml",
  detection_only: true,
  enable_jersey: true,
  jersey_backend: "best_temporal",
  jersey_interval_frames: 24,
  live_preview: true,
  preview_interval_frames: 15,
};

function pct(value: number | undefined) { return `${Number(value ?? 0).toFixed(1)}%`; }
function metric(value: number | undefined, digits = 2) { return Number(value ?? 0).toFixed(digits); }

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
  const [detectionOnly, setDetectionOnly] = useState(FALLBACK.detection_only);
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
    setVideoSource(d.video_source); setModel(d.model); setConfidence(d.detector_conf);
    setInferenceConf(d.inference_conf); setBallConf(d.ball_conf); setImgsz(d.imgsz);
    setDevice(d.device); setTracker(d.tracker); setDetectionOnly(d.detection_only);
    setEnableJersey(d.enable_jersey); setJerseyInterval(d.jersey_interval_frames);
    setLivePreview(d.live_preview); setPreviewInterval(d.preview_interval_frames);
  }

  function applyCanonicalMatchConfig(d: LabDefaults | undefined) {
    if (!d?.match_config) return;
    setMatchConfigText(JSON.stringify(d.match_config, null, 2));
  }

  async function refreshHealth() {
    try {
      const response = await fetch("/api/lab/health", { cache: "no-store" });
      const data: WorkerHealth = await response.json();
      setHealth(data);
      if (data.status === "ok" && data.lab_defaults) {
        if (!defaultsApplied.current) {
          applyDefaults(data.lab_defaults); defaultsApplied.current = true;
        }
        if (data.lab_defaults.match_config && !matchConfigApplied.current) {
          applyCanonicalMatchConfig(data.lab_defaults); matchConfigApplied.current = true;
        }
      }
    } catch (error) {
      setHealth({ status: "error", error: error instanceof Error ? error.message : String(error) });
    }
  }

  useEffect(() => { void refreshHealth(); }, []);

  useEffect(() => {
    if (!job?.job_id || ["completed", "failed"].includes(job.status ?? "")) return;
    const jobId = job.job_id; let cancelled = false;
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
    void poll(); const timer = window.setInterval(() => void poll(), 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [job?.job_id, job?.status, livePreview]);

  const parsedMatchConfig = useMemo(() => {
    try { return { value: JSON.parse(matchConfigText), error: "" }; }
    catch (error) { return { value: null, error: error instanceof Error ? error.message : String(error) }; }
  }, [matchConfigText]);

  async function submitJob() {
    setFormError(""); setLogs([]);
    if (!parsedMatchConfig.value) { setFormError(`Invalid match_config.json: ${parsedMatchConfig.error}`); return; }
    if (endTime <= startTime) { setFormError("End time must be greater than start time."); return; }
    if (inferenceConf > Math.min(confidence, ballConf)) {
      setFormError("YOLO inference confidence must be <= player and ball confidence."); return;
    }
    setSubmitting(true);
    try {
      const response = await fetch("/api/lab/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_source: videoSource, start_time_sec: startTime, end_time_sec: endTime,
          model, detector_conf: confidence, inference_conf: inferenceConf, ball_conf: ballConf,
          imgsz, device, tracker, detection_only: detectionOnly, enable_jersey: enableJersey,
          jersey_backend: health.lab_defaults?.jersey_backend ?? "best_temporal",
          jersey_interval_frames: jerseyInterval,
          live_preview: livePreview, preview_interval_frames: previewInterval,
          single_forward: true, smart_ball_mode: "off",
          diagnostics_batch_size: 100, match_config: parsedMatchConfig.value,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail ?? data.error ?? `HTTP ${response.status}`);
      setJob(data); setPreviewNonce(Date.now());
    } catch (error) { setFormError(error instanceof Error ? error.message : String(error)); }
    finally { setSubmitting(false); }
  }

  const summary = job?.summary ?? null;
  const players = summary?.players; const ball = summary?.ball; const performance = summary?.performance;
  const processingFps = performance?.analysis_fps ?? (summary?.total_frames && summary?.elapsed_sec ? summary.total_frames / summary.elapsed_sec : 0);
  const analysisActive = ["queued", "running"].includes(job?.status ?? "");
  const workerLabel = health.status === "ok" ? "online" : health.status ? "offline" : "checking";
  const previewUrl = job?.job_id ? `/api/lab/jobs/${job.job_id}/preview?t=${previewNonce}` : "";
  const videoUrl = job?.job_id ? `/api/lab/jobs/${job.job_id}/preview-video` : "";

  return (
    <main>
      <header>
        <div>
          <h1>CourtFlux Beta Lab</h1>
          <p>P1 Identity QA · canonical model v1.1 · single-forward ON · Smart Ball OFF · Supabase persistence.</p>
        </div>
        <button className={`badge ${health.status === "ok" ? "ok" : health.status ? "error" : ""}`} onClick={() => void refreshHealth()} type="button" aria-label={`Refresh worker status. Worker is ${workerLabel}.`}>
          <span aria-live="polite">Worker: {workerLabel}{health.devices?.mps ? " · MPS" : health.devices?.cuda ? " · CUDA" : health.status === "ok" ? " · CPU" : ""}{health.supabase_persistence ? " · Supabase" : ""}</span>
        </button>
      </header>

      <div className="grid">
        <section className="panel span-8" aria-labelledby="analysis-run-title">
          <div className="status-line"><h2 id="analysis-run-title" style={{ margin: 0 }}>Analysis run</h2><span className="muted">Contract: {model} · {imgsz}px · P {confidence.toFixed(2)} / infer {inferenceConf.toFixed(2)} / ball {ballConf.toFixed(2)}</span></div>
          <div className="form-grid">
            <div className="field full"><label htmlFor="video-source">YouTube URL / worker-local video path</label><input id="video-source" value={videoSource} onChange={(e) => setVideoSource(e.target.value)} /></div>
            <div className="field"><label htmlFor="start-time">Start (seconds)</label><input id="start-time" type="number" min="0" step="1" value={startTime} onChange={(e) => setStartTime(Number(e.target.value))} /></div>
            <div className="field"><label htmlFor="end-time">End (seconds)</label><input id="end-time" type="number" min="1" step="1" value={endTime} onChange={(e) => setEndTime(Number(e.target.value))} /></div>
            <div className="field full"><label htmlFor="model-path">Model path on worker</label><input id="model-path" value={model} onChange={(e) => setModel(e.target.value)} /><span className="muted">Canonical: weights/model_v1.1.0.pt</span></div>
            <div className="field"><label htmlFor="player-confidence">Player confidence</label><input id="player-confidence" type="number" min="0.01" max="0.95" step="0.01" value={confidence} onChange={(e) => setConfidence(Number(e.target.value))} /></div>
            <div className="field"><label htmlFor="inference-confidence">YOLO inference confidence</label><input id="inference-confidence" type="number" min="0.001" max="0.95" step="0.01" value={inferenceConf} onChange={(e) => setInferenceConf(Number(e.target.value))} /></div>
            <div className="field"><label htmlFor="ball-confidence">Ball acceptance confidence</label><input id="ball-confidence" type="number" min="0.001" max="0.95" step="0.01" value={ballConf} onChange={(e) => setBallConf(Number(e.target.value))} /></div>
            <div className="field"><label htmlFor="image-size">Image size</label><select id="image-size" value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))}>{[640, 768, 960, 1280].map((size) => <option key={size} value={size}>{size}</option>)}</select></div>
            <div className="field"><label htmlFor="device">Device</label><select id="device" value={device} onChange={(e) => setDevice(e.target.value as "mps" | "cpu" | "cuda")}><option value="mps">MPS</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></div>
            <div className="field"><label htmlFor="tracker">Tracker</label><select id="tracker" value={tracker} onChange={(e) => setTracker(e.target.value as "botsort.yaml" | "bytetrack.yaml")}><option value="botsort.yaml">BoT-SORT</option><option value="bytetrack.yaml">ByteTrack</option></select></div>
            <div className="field"><label htmlFor="detection-only"><input id="detection-only" type="checkbox" checked={detectionOnly} onChange={(e) => setDetectionOnly(e.target.checked)} /> Detection-only mode</label><span className="muted">Matches the local performance/diagnostic Lab path.</span></div>
            <div className="field"><label htmlFor="enable-jersey"><input id="enable-jersey" type="checkbox" checked={enableJersey} onChange={(e) => setEnableJersey(e.target.checked)} /> Jersey recognition</label><span className="muted">Keep ON for the P1 Identity exit gate.</span></div>
            <div className="field"><label htmlFor="jersey-interval">Jersey interval (frames)</label><input id="jersey-interval" type="number" min="1" max="300" value={jerseyInterval} disabled={!enableJersey} onChange={(e) => setJerseyInterval(Number(e.target.value))} /><span className="muted">Diagnostic default: 24; best crops are still collected every eligible frame.</span></div>
            <div className="field"><span>Persistence</span><strong>Supabase only</strong><span className="muted">Worker-backed runs never write local tracking.db.</span></div>
            <div className="field"><label htmlFor="live-preview"><input id="live-preview" type="checkbox" checked={livePreview} disabled={!detectionOnly} onChange={(e) => setLivePreview(e.target.checked)} /> Live diagnostic preview + video</label><span className="muted">Same annotated renderer as localhost.</span></div>
            <div className="field"><label htmlFor="preview-interval">Preview interval (frames)</label><input id="preview-interval" type="number" min="1" max="300" value={previewInterval} disabled={!livePreview || !detectionOnly} onChange={(e) => setPreviewInterval(Number(e.target.value))} /></div>
          </div>

          <div className="actions">
            <button className="primary" type="button" onClick={() => void submitJob()} disabled={submitting || health.status !== "ok" || analysisActive} aria-busy={submitting || analysisActive}>{submitting ? "Submitting…" : analysisActive ? "Analysis running…" : "Run analysis"}</button>
            <button type="button" onClick={() => applyDefaults(health.lab_defaults ?? FALLBACK)} disabled={analysisActive}>Reset to worker defaults</button>
            <span className="muted">P1 defaults: v1.1.0 · BoT-SORT · best_temporal · Jersey ON · interval 24 · single-forward ON · Smart Ball OFF · Supabase only.</span>
          </div>
          {formError ? <div className="error-box" role="alert" style={{ marginTop: 14 }}>{formError}</div> : null}
        </section>

        <section className="panel span-4" aria-labelledby="match-config-title">
          <div className="status-line"><h2 id="match-config-title" style={{ margin: 0 }}>match_config.json</h2><button type="button" onClick={() => applyCanonicalMatchConfig(health.lab_defaults)} disabled={!health.lab_defaults?.match_config || analysisActive}>Reload from code</button></div>
          <p className="muted">Loaded from the worker canonical {health.lab_defaults?.match_config_source ?? "data/match_config.json"}. Edit it here for this run; the submitted JSON is snapshotted into the job.</p>
          <textarea value={matchConfigText} onChange={(e) => setMatchConfigText(e.target.value)} spellCheck={false} aria-label="Match configuration JSON" aria-invalid={Boolean(parsedMatchConfig.error)} />
          {parsedMatchConfig.error ? <div className="error-box" role="alert">{parsedMatchConfig.error}</div> : <div className="muted" role="status">JSON valid · canonical source loaded from worker</div>}
        </section>

        {job?.job_id && livePreview ? <section className="panel span-12" aria-labelledby="preview-title">
          <div className="status-line"><h2 id="preview-title" style={{ margin: 0 }}>Visual QA</h2><span className="muted">Sampled diagnostic overlays; visual evidence, not a quality metric.</span></div>
          {analysisActive || job.artifacts?.preview ? <img src={previewUrl} alt="Live CourtFlux diagnostic preview" style={{ width: "100%", maxHeight: 680, objectFit: "contain", marginTop: 16, background: "#080b10" }} /> : null}
          {job.status === "completed" && job.artifacts?.analysis_video ? <div style={{ marginTop: 18 }}><h3>Analysis video preview</h3><video controls src={videoUrl} style={{ width: "100%", maxHeight: 680, background: "#080b10" }} /><p><a href={videoUrl} download>Download analysis_preview.mp4</a></p></div> : null}
        </section> : null}

        <section className="panel span-12" aria-labelledby="results-title" aria-busy={analysisActive}>
          <div className="status-line"><h2 id="results-title" style={{ margin: 0 }}>Results</h2><span role="status" aria-live="polite">{job?.status ? <span className={`status ${job.status}`}>{job.status}</span> : <span className="muted">No remote run submitted yet.</span>}</span>{job?.job_id ? <span className="muted">Job {job.job_id}</span> : null}</div>
          {job?.error ? <div className="error-box" role="alert" style={{ marginTop: 14 }}>{job.error}</div> : null}
          {summary ? <>
            <p className="muted">Run {summary.run_id ?? "—"} · {summary.model ?? model} · {summary.imgsz ?? imgsz}px · infer/player/ball {summary.inference_conf ?? inferenceConf} / {summary.player_conf ?? confidence} / {summary.ball_conf ?? ballConf}</p>
            <div className="metrics" style={{ marginTop: 18 }}>
              <div className="metric"><div className="label">Gameplay frames</div><div className="value">{summary.gameplay_frames ?? 0}</div></div>
              <div className="metric"><div className="label">Analysis FPS</div><div className="value">{metric(processingFps)}</div></div>
              <div className="metric"><div className="label">Real-time factor</div><div className="value">{metric(performance?.realtime_factor)}x</div></div>
              <div className="metric"><div className="label">Players / frame</div><div className="value">{metric(players?.parsed_per_gameplay_frame, 3)}</div></div>
              <div className="metric"><div className="label">Team classified</div><div className="value">{pct(players?.team_classification_rate)}</div></div>
              <div className="metric"><div className="label">Jersey identified</div><div className="value">{pct(players?.jersey_identification_rate)}</div></div>
              <div className="metric"><div className="label">Ball candidate frames</div><div className="value">{pct(ball?.raw_candidate_frame_rate)}</div></div>
              <div className="metric"><div className="label">Ball selected</div><div className="value">{pct(ball?.selected_detection_frame_rate)}</div></div>
              <div className="metric"><div className="label">Ball missing</div><div className="value">{pct(ball?.missing_frame_rate)}</div></div>
              <div className="metric"><div className="label">Ball mean confidence</div><div className="value">{metric(ball?.selected_conf_mean ?? undefined, 3)}</div></div>
              <div className="metric"><div className="label">Jersey attempts</div><div className="value">{players?.jersey_attempts ?? 0}</div></div>
              <div className="metric"><div className="label">Elapsed</div><div className="value">{metric(summary.elapsed_sec, 1)}s</div></div>
            </div>
            {summary.extra?.identity_debug ? <details open style={{ marginTop: 18 }}><summary className="muted">Identity debug probe</summary><div style={{ marginTop: 10 }}><pre>{JSON.stringify(summary.extra.identity_debug, null, 2)}</pre></div></details> : null}
            {ball?.continuity ? <details open style={{ marginTop: 18 }}><summary className="muted">Ball continuity</summary><div style={{ marginTop: 10 }}><pre>{JSON.stringify(ball.continuity, null, 2)}</pre></div></details> : null}
            {summary.detector_timing ? <details style={{ marginTop: 18 }}><summary className="muted">Detector timing</summary><div style={{ marginTop: 10 }}><pre>{JSON.stringify(summary.detector_timing, null, 2)}</pre></div></details> : null}
            {performance?.stages ? <details style={{ marginTop: 18 }}><summary className="muted">Pipeline timing</summary><div style={{ marginTop: 10 }}><pre>{JSON.stringify(performance.stages, null, 2)}</pre></div></details> : null}
            {players?.team_counts ? <details style={{ marginTop: 18 }}><summary className="muted">Team detection counts</summary><div style={{ marginTop: 10 }}><pre>{JSON.stringify(players.team_counts, null, 2)}</pre></div></details> : null}
            <details style={{ marginTop: 18 }}><summary className="muted">Raw summary JSON</summary><div style={{ marginTop: 10 }}><pre>{JSON.stringify(summary, null, 2)}</pre></div></details>
          </> : null}
        </section>

        <section className="panel span-12" aria-labelledby="worker-log-title"><h2 id="worker-log-title">Worker log</h2><pre aria-label="Worker processing log">{logs.length ? logs.join("\n") : "Logs will appear here while the worker processes the job."}</pre></section>
      </div>
    </main>
  );
}
