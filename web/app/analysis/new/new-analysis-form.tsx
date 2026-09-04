"use client";

import Image from "next/image";
import Link from "next/link";
import { useMemo, useState } from "react";
import { createClient } from "@/lib/supabase/client";
import styles from "./page.module.css";

type TeamOption = {
  id: string;
  name: string;
  shortName: string | null;
};

type Props = {
  workspaceName: string;
  role: string;
  planName: string;
  usedMinutes: number;
  allowance: number | null;
  usageLevel: "healthy" | "warning" | "exhausted";
  teams: TeamOption[];
};

type UploadTicket = {
  bucket: string;
  path: string;
  token: string;
  source: string;
  retention_days?: number;
  expires_at?: string;
  error?: string;
};

type JobResult = {
  job_id?: string;
  run_id?: string;
  status?: string;
  error?: string;
  detail?: string;
  code?: string;
};

const MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024;

function formatMinutes(value: number) {
  if (!Number.isFinite(value)) return "0";
  if (value < 10) return value.toFixed(1).replace(/\.0$/, "");
  return Math.round(value).toString();
}

function parseColors(value: string) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseRoster(value: string) {
  const roster: Record<string, { name: string }> = {};
  for (const rawLine of value.split(/\n+/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = line.match(/^#?(\d{1,3})(?:\s*[-,:]\s*|\s+)?(.*)$/);
    if (!match) continue;
    const jersey = match[1];
    const name = match[2].trim() || `Player #${jersey}`;
    roster[jersey] = { name };
  }
  return roster;
}

function videoDuration(file: File) {
  return new Promise<number>((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const duration = video.duration;
      URL.revokeObjectURL(url);
      if (Number.isFinite(duration) && duration > 0) resolve(duration);
      else reject(new Error("Could not read video duration"));
    };
    video.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read video metadata"));
    };
    video.src = url;
  });
}

export function NewAnalysisForm({
  workspaceName,
  role,
  planName,
  usedMinutes,
  allowance,
  usageLevel,
  teams,
}: Props) {
  const [sourceMode, setSourceMode] = useState<"upload" | "youtube">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [startMinutes, setStartMinutes] = useState("0");
  const [endMinutes, setEndMinutes] = useState("60");
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [gameDate, setGameDate] = useState("");
  const [competition, setCompetition] = useState("");
  const [homeColors, setHomeColors] = useState("");
  const [awayColors, setAwayColors] = useState("");
  const [homeRoster, setHomeRoster] = useState("");
  const [awayRoster, setAwayRoster] = useState("");
  const [enableJersey, setEnableJersey] = useState(false);
  const [rightsConfirmed, setRightsConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<JobResult | null>(null);

  const start = Number(startMinutes);
  const end = Number(endMinutes);
  const estimatedMinutes = Number.isFinite(start) && Number.isFinite(end) && end > start ? end - start : 0;
  const remaining = allowance === null ? null : Math.max(0, allowance - usedMinutes);
  const remainingAfter = remaining === null ? null : remaining - estimatedMinutes;
  const wouldExceed = remainingAfter !== null && remainingAfter < -0.01;
  const projectedPercent = allowance && allowance > 0 ? ((usedMinutes + estimatedMinutes) / allowance) * 100 : 0;
  const canManage = ["owner", "admin", "analyst"].includes(role);

  const sourceReady = sourceMode === "upload" ? Boolean(file) : /^https?:\/\//i.test(youtubeUrl.trim());
  const contextReady = homeTeam.trim().length > 0 && awayTeam.trim().length > 0 && homeTeam.trim() !== awayTeam.trim();
  const scopeReady = estimatedMinutes > 0;
  const blocked = !canManage || usageLevel === "exhausted" || wouldExceed;
  const canSubmit = sourceReady && contextReady && scopeReady && rightsConfirmed && !blocked && !busy;

  const teamNames = useMemo(() => teams.map((team) => team.name), [teams]);

  async function chooseFile(nextFile: File | null) {
    setError("");
    setResult(null);
    setFile(nextFile);
    if (!nextFile) return;
    if (nextFile.size > MAX_FILE_BYTES) {
      setFile(null);
      setError("Video uploads are limited to 2 GB in the private beta.");
      return;
    }
    try {
      const durationSeconds = await videoDuration(nextFile);
      setStartMinutes("0");
      setEndMinutes((durationSeconds / 60).toFixed(1));
    } catch {
      // Keep the editable 60-minute default when browser metadata is unavailable.
    }
  }

  async function uploadVideo(selectedFile: File) {
    setStage("Preparing secure upload…");
    const ticketResponse = await fetch("/api/uploads/video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_name: selectedFile.name,
        file_size: selectedFile.size,
        content_type: selectedFile.type,
      }),
    });
    const ticket = (await ticketResponse.json()) as UploadTicket;
    if (!ticketResponse.ok || !ticket.token) {
      throw new Error(ticket.error || "Could not prepare the video upload");
    }

    setStage("Uploading video securely…");
    const supabase = createClient();
    const { error: uploadError } = await supabase.storage
      .from(ticket.bucket)
      .uploadToSignedUrl(ticket.path, ticket.token, selectedFile, {
        contentType: selectedFile.type || "video/mp4",
      });
    if (uploadError) throw uploadError;
    return ticket.source;
  }

  async function submitAnalysis() {
    setError("");
    setResult(null);
    if (!canSubmit) return;

    setBusy(true);
    try {
      const cleanHome = homeTeam.trim();
      const cleanAway = awayTeam.trim();
      let source = youtubeUrl.trim();
      if (sourceMode === "upload") {
        if (!file) throw new Error("Choose a basketball video first");
        source = await uploadVideo(file);
      }

      const homeKit = parseColors(homeColors);
      const awayKit = parseColors(awayColors);
      const matchConfig = {
        metadata: {
          home_team: cleanHome,
          away_team: cleanAway,
          active_kits: {
            [cleanHome]: "current",
            [cleanAway]: "current",
          },
        },
        teams: {
          [cleanHome]: {
            kits: { current: homeKit },
            roster: parseRoster(homeRoster),
          },
          [cleanAway]: {
            kits: { current: awayKit },
            roster: parseRoster(awayRoster),
          },
        },
      };

      setStage("Queuing CourtFlux analysis…");
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_source: source,
          start_time_sec: start * 60,
          end_time_sec: end * 60,
          model: "weights/Model_v0.1.pt",
          detector_conf: 0.1,
          imgsz: 960,
          device: "cuda",
          tracker: "botsort.yaml",
          detection_only: true,
          enable_jersey: enableJersey,
          jersey_interval_frames: 12,
          persist_results: true,
          db_batch_size: 100,
          diagnostics_batch_size: 100,
          match_config: matchConfig,
          game_context: {
            home_team: cleanHome,
            away_team: cleanAway,
            game_date: gameDate || null,
            competition: competition.trim() || null,
            source_type: sourceMode,
            source_label: sourceMode === "upload" ? file?.name ?? null : "YouTube",
            home_kit_colors: homeKit,
            away_kit_colors: awayKit,
          },
        }),
      });
      const data = (await response.json()) as JobResult;
      if (!response.ok) {
        throw new Error(data.detail || data.error || `Could not start analysis (HTTP ${response.status})`);
      }
      setResult(data);
      setStage("");
    } catch (submitError) {
      setStage("");
      setError(submitError instanceof Error ? submitError.message : String(submitError));
    } finally {
      setBusy(false);
    }
  }

  if (result?.job_id) {
    return (
      <main className={styles.shell}>
        <header className={styles.topbar}>
          <Link href="/" className={styles.brand} aria-label="CourtFlux home">
            <Image src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux" width={660} height={185} priority />
          </Link>
          <Link className={styles.accountLink} href="/account">Back to dashboard</Link>
        </header>
        <section className={styles.successCard} role="status" aria-live="polite">
          <span className={styles.successMark}>✓</span>
          <div className={styles.eyebrow}>Analysis queued</div>
          <h1>Your basketball video is in the CourtFlux pipeline.</h1>
          <p>
            CourtFlux now has a dedicated processing page for this run. You can follow the real queue and worker state there, leave the page, and return later from your workspace.
          </p>
          <div className={styles.successMeta}>
            <span>Estimated usage</span><strong>{formatMinutes(estimatedMinutes)} min</strong>
            <span>Job</span><strong>{result.job_id.slice(0, 8)}…</strong>
          </div>
          <div className={styles.successActions}>
            <Link className={styles.primaryButton} href={result.run_id ? `/analysis/${result.run_id}` : "/account"}>Track analysis</Link>
            <Link className={styles.secondaryButton} href="/account">Dashboard</Link>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className={styles.shell}>
      <header className={styles.topbar}>
        <Link href="/" className={styles.brand} aria-label="CourtFlux home">
          <Image src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux" width={660} height={185} priority />
        </Link>
        <div className={styles.topActions}>
          <span>{workspaceName}</span>
          <Link className={styles.accountLink} href="/account">Dashboard</Link>
        </div>
      </header>

      <section className={styles.intro}>
        <div>
          <div className={styles.eyebrow}>New analysis</div>
          <h1>Turn your next basketball video into structured evidence.</h1>
          <p>
            Add the footage, identify the two teams, choose the part of the video to process and review the analysis-minute estimate before it enters the pipeline.
          </p>
        </div>
        <aside className={styles.allowanceCard} aria-label="Analysis allowance">
          <span>{planName} plan</span>
          <strong>{remaining === null ? "—" : `${formatMinutes(remaining)} min`}</strong>
          <small>{allowance === null ? "Allowance not configured" : `remaining from ${formatMinutes(allowance)} min this month`}</small>
        </aside>
      </section>

      <div className={styles.layout}>
        <div className={styles.formStack}>
          <section className={styles.panel} aria-labelledby="source-title">
            <div className={styles.sectionHeading}><span>1</span><div><h2 id="source-title">Choose your video</h2><p>Upload game film or analyse a YouTube recording you are allowed to use.</p></div></div>
            <div className={styles.segmented} aria-label="Video source">
              <button type="button" className={sourceMode === "upload" ? styles.activeSegment : ""} onClick={() => setSourceMode("upload")}>Upload video</button>
              <button type="button" className={sourceMode === "youtube" ? styles.activeSegment : ""} onClick={() => setSourceMode("youtube")}>YouTube URL</button>
            </div>

            {sourceMode === "upload" ? (
              <>
                <label className={styles.uploadBox} htmlFor="video-file">
                  <input id="video-file" type="file" accept="video/mp4,video/quicktime,video/webm,video/x-m4v,.mp4,.mov,.webm,.m4v" onChange={(event) => void chooseFile(event.target.files?.[0] ?? null)} />
                  <strong>{file ? file.name : "Select basketball video"}</strong>
                  <span>{file ? `${(file.size / (1024 * 1024)).toFixed(1)} MB` : "MP4, MOV, WEBM or M4V · up to 2 GB"}</span>
                </label>
                <div className={styles.scopeHint}>
                  <strong>60-day video retention.</strong> CourtFlux automatically deletes the uploaded source video 60 days after upload. Your derived analysis results and statistics remain in the workspace. <Link href="/video-data">How video retention works</Link>.
                </div>
              </>
            ) : (
              <div className={styles.field}>
                <label htmlFor="youtube-url">YouTube URL</label>
                <input id="youtube-url" type="url" placeholder="https://www.youtube.com/watch?v=…" value={youtubeUrl} onChange={(event) => setYoutubeUrl(event.target.value)} />
                <small>CourtFlux downloads the video on the private worker before analysis. The original YouTube video remains controlled by YouTube and its uploader.</small>
              </div>
            )}
          </section>

          <section className={styles.panel} aria-labelledby="context-title">
            <div className={styles.sectionHeading}><span>2</span><div><h2 id="context-title">Match context</h2><p>Only the two team names are required. Everything else can be added later.</p></div></div>
            <datalist id="workspace-teams">{teamNames.map((name) => <option value={name} key={name} />)}</datalist>
            <div className={styles.twoCols}>
              <div className={styles.field}><label htmlFor="home-team">Home team *</label><input id="home-team" list="workspace-teams" value={homeTeam} onChange={(event) => setHomeTeam(event.target.value)} placeholder="Home team" /></div>
              <div className={styles.field}><label htmlFor="away-team">Away team *</label><input id="away-team" list="workspace-teams" value={awayTeam} onChange={(event) => setAwayTeam(event.target.value)} placeholder="Away team" /></div>
              <div className={styles.field}><label htmlFor="game-date">Game date</label><input id="game-date" type="date" value={gameDate} onChange={(event) => setGameDate(event.target.value)} /></div>
              <div className={styles.field}><label htmlFor="competition">Competition</label><input id="competition" value={competition} onChange={(event) => setCompetition(event.target.value)} placeholder="League, tournament or friendly" /></div>
              <div className={styles.field}><label htmlFor="home-colors">Home-team kit colours</label><input id="home-colors" value={homeColors} onChange={(event) => setHomeColors(event.target.value)} placeholder="Blue, white" /><small>Optional, but useful for team classification.</small></div>
              <div className={styles.field}><label htmlFor="away-colors">Away-team kit colours</label><input id="away-colors" value={awayColors} onChange={(event) => setAwayColors(event.target.value)} placeholder="Red, black" /><small>Use simple colour names separated by commas.</small></div>
            </div>

            <details className={styles.optionalDetails}>
              <summary>Optional jersey numbers / roster</summary>
              <p>You can analyse anonymous tracks without a roster. Add one player per line only if you have the information.</p>
              <div className={styles.twoCols}>
                <div className={styles.field}><label htmlFor="home-roster">{homeTeam.trim() || "Home team"} roster</label><textarea id="home-roster" value={homeRoster} onChange={(event) => setHomeRoster(event.target.value)} placeholder={"7 Jordan Smith\n12 Alex Brown"} /></div>
                <div className={styles.field}><label htmlFor="away-roster">{awayTeam.trim() || "Away team"} roster</label><textarea id="away-roster" value={awayRoster} onChange={(event) => setAwayRoster(event.target.value)} placeholder={"4 Sam Lee\n21 Chris Jones"} /></div>
              </div>
              <label className={styles.checkbox}><input type="checkbox" checked={enableJersey} onChange={(event) => setEnableJersey(event.target.checked)} /> Try experimental jersey-number recognition</label>
            </details>
          </section>

          <section className={styles.panel} aria-labelledby="scope-title">
            <div className={styles.sectionHeading}><span>3</span><div><h2 id="scope-title">Choose the analysis window</h2><p>Analysis minutes are based on processed video duration.</p></div></div>
            <div className={styles.twoCols}>
              <div className={styles.field}><label htmlFor="start-minutes">Start at (minutes)</label><input id="start-minutes" type="number" min="0" step="0.1" value={startMinutes} onChange={(event) => setStartMinutes(event.target.value)} /></div>
              <div className={styles.field}><label htmlFor="end-minutes">End at (minutes)</label><input id="end-minutes" type="number" min="0.1" step="0.1" value={endMinutes} onChange={(event) => setEndMinutes(event.target.value)} /></div>
            </div>
            <div className={styles.scopeHint}>A 42-minute game segment uses approximately <strong>42 analysis minutes</strong>. Processing speed does not change the allowance.</div>
          </section>
        </div>

        <aside className={styles.review} aria-labelledby="review-title">
          <div className={styles.eyebrow}>Review</div>
          <h2 id="review-title">Ready to analyse?</h2>
          <div className={styles.reviewRows}>
            <div><span>Source</span><strong>{sourceMode === "upload" ? file?.name || "Not selected" : youtubeUrl.trim() ? "YouTube" : "Not added"}</strong></div>
            <div><span>Match</span><strong>{contextReady ? `${homeTeam.trim()} vs ${awayTeam.trim()}` : "Add both teams"}</strong></div>
            <div><span>Analysis window</span><strong>{scopeReady ? `${formatMinutes(start)}–${formatMinutes(end)} min` : "Check start/end"}</strong></div>
            <div><span>Estimated usage</span><strong className={styles.orange}>{formatMinutes(estimatedMinutes)} min</strong></div>
            <div><span>Remaining after</span><strong>{remainingAfter === null ? "—" : `${formatMinutes(Math.max(0, remainingAfter))} min`}</strong></div>
          </div>

          <label className={styles.checkbox}>
            <input type="checkbox" checked={rightsConfirmed} onChange={(event) => setRightsConfirmed(event.target.checked)} />
            <span>I confirm I am allowed to submit this footage for analysis and understand that direct uploads are deleted after 60 days while derived analysis results and statistics are retained.</span>
          </label>
          <p className={styles.privateNote}>Read the <Link href="/video-data">Video & Data Policy</Link> and <Link href="/terms">private-beta terms</Link>.</p>

          {projectedPercent >= 80 && !wouldExceed && usageLevel !== "exhausted" ? <div className={styles.warningBox}>This analysis will take the workspace above 80% of its monthly allowance.</div> : null}
          {wouldExceed ? <div className={styles.errorBox}>This analysis is longer than your remaining allowance. Shorten the analysis window or ask a CourtFlux admin for additional beta minutes.</div> : null}
          {usageLevel === "exhausted" ? <div className={styles.errorBox}>Your monthly allowance is currently exhausted. New analyses are paused until reset or a manual beta top-up.</div> : null}
          {!canManage ? <div className={styles.errorBox}>Your workspace role can view analyses but cannot start a new one.</div> : null}
          {error ? <div className={styles.errorBox} role="alert">{error}</div> : null}
          {stage ? <div className={styles.stageBox} role="status" aria-live="polite">{stage}</div> : null}

          <button className={styles.primaryButton} type="button" disabled={!canSubmit} onClick={() => void submitAnalysis()} aria-busy={busy}>
            {busy ? "Preparing analysis…" : `Start analysis · ${formatMinutes(estimatedMinutes)} min`}
          </button>
          <p className={styles.privateNote}>Billing is not active in the private beta. CourtFlux will not charge an overage automatically.</p>
          <Link className={styles.secondaryLink} href="/account">Cancel and return to dashboard</Link>
        </aside>
      </div>
    </main>
  );
}
