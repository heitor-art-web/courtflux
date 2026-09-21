import Link from "next/link";
import styles from "./game-story.module.css";

type UnknownRecord = Record<string, unknown>;

type GameStoryProps = {
  summary: UnknownRecord | null;
  runId: string;
  startTimeSec: number | null;
  endTimeSec: number | null;
};

function record(value: unknown): UnknownRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as UnknownRecord
    : {};
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function count(value: unknown) {
  return Math.max(0, Math.round(number(value) ?? 0));
}

function percent(value: unknown) {
  const raw = number(value);
  if (raw === null) return null;
  const normalized = raw >= 0 && raw <= 1 ? raw * 100 : raw;
  return Math.max(0, Math.min(100, normalized));
}

function percentText(value: unknown) {
  const resolved = percent(value);
  return resolved === null ? "—" : `${resolved.toFixed(1)}%`;
}

function formatClock(seconds: number | null) {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${minutes}:${String(secs).padStart(2, "0")}`;
}

function plural(value: number, singular: string, pluralValue = `${singular}s`) {
  return `${value} ${value === 1 ? singular : pluralValue}`;
}

function humanize(value: string) {
  if (value === "unattributed") return "Unattributed";
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function metricFromEventCounts(eventCounts: UnknownRecord, key: string) {
  return count(eventCounts[key]);
}

export function GameStory({ summary, runId, startTimeSec, endTimeSec }: GameStoryProps) {
  if (!summary) return null;

  const p2 = record(summary.p2);
  if (!Object.keys(p2).length) {
    return (
      <section className={styles.storyPanel} aria-labelledby="game-story-title">
        <div className={styles.eyebrow}>Game story</div>
        <div className={styles.legacyGrid}>
          <div>
            <h2 id="game-story-title">This run predates CourtFlux game-story metrics.</h2>
            <p>
              This completed analysis contains tracking diagnostics, but it does not contain the canonical P2
              possession and event summary used by current runs. CourtFlux will not invent passes, turnovers,
              rebounds or shot outcomes from data that was never produced.
            </p>
          </div>
          <div className={styles.legacyAction}>
            <strong>For a game narrative</strong>
            <span>Run the clip again with the current local pipeline.</span>
            <Link href="/analysis/new">Start a current analysis</Link>
          </div>
        </div>
      </section>
    );
  }

  const events = record(p2.events);
  const eventCounts = record(events.counts);
  const possession = record(p2.possession);
  const transitions = record(p2.transitions);
  const shots = record(p2.shots);
  const outcomes = record(shots.outcomes);
  const teams = record(p2.teams);

  const passes = count(p2.passes) || count(transitions.passes) || metricFromEventCounts(eventCounts, "pass_candidate");
  const turnovers = count(p2.turnovers) || count(transitions.turnovers) || metricFromEventCounts(eventCounts, "turnover_candidate");
  const rebounds = count(p2.rebounds) || metricFromEventCounts(eventCounts, "rebound_candidate");
  const shotAttempts = count(shots.attempts) || metricFromEventCounts(eventCounts, "shot_attempt");
  const madeShots = count(outcomes.made);
  const missedShots = count(outcomes.missed);
  const unknownShots = count(outcomes.unknown);
  const totalEvents = count(events.total);
  const possessionRate = percent(possession.possessed_rate);
  const actorAttribution = percent(events.actor_attribution_rate);
  const teamAttribution = percent(events.team_attribution_rate);
  const possessionConfidence = percent(possession.avg_possession_confidence);
  const unknownShotRate = percent(shots.unknown_outcome_rate);
  const analysisDuration = typeof startTimeSec === "number" && typeof endTimeSec === "number" && endTimeSec > startTimeSec
    ? formatClock(endTimeSec - startTimeSec)
    : null;

  const narrativeParts = [
    passes ? plural(passes, "pass candidate") : null,
    turnovers ? plural(turnovers, "turnover candidate") : null,
    rebounds ? plural(rebounds, "rebound candidate") : null,
    shotAttempts ? plural(shotAttempts, "shot attempt") : null,
  ].filter((value): value is string => Boolean(value));

  const teamRows = Object.entries(teams)
    .map(([teamId, raw]) => ({ teamId, stats: record(raw) }))
    .filter(({ stats }) => Object.keys(stats).length > 0)
    .sort((left, right) => (number(right.stats.possession_share_rate) ?? 0) - (number(left.stats.possession_share_rate) ?? 0));

  return (
    <section className={styles.storyPanel} aria-labelledby="game-story-title">
      <div className={styles.storyHeader}>
        <div>
          <div className={styles.eyebrow}>Game story</div>
          <h2 id="game-story-title">What happened in this analysed window</h2>
          <p className={styles.storyLead}>
            {analysisDuration ? `Across ${analysisDuration} of analysed video, ` : "In this analysed window, "}
            CourtFlux&apos;s canonical game layer produced {plural(totalEvents, "event")}.
            {narrativeParts.length ? ` The evidence includes ${narrativeParts.join(", ")}.` : " No reviewable pass, turnover, rebound or shot events were emitted."}
            {possessionRate !== null ? ` Ball control was established on ${possessionRate.toFixed(1)}% of P2-eligible frames.` : ""}
          </p>
        </div>
        <div className={styles.storyStamp}>Canonical P2</div>
      </div>

      <div className={styles.storyMetrics} aria-label="Game story metrics">
        <article><span>Ball control</span><strong>{possessionRate === null ? "—" : `${possessionRate.toFixed(1)}%`}</strong><small>P2-eligible frames marked possessed</small></article>
        <article><span>Passes</span><strong>{passes}</strong><small>Pass candidates</small></article>
        <article><span>Turnovers</span><strong>{turnovers}</strong><small>Turnover candidates</small></article>
        <article><span>Rebounds</span><strong>{rebounds}</strong><small>Rebound candidates</small></article>
        <article><span>Shots</span><strong>{shotAttempts}</strong><small>{madeShots} made · {missedShots} missed · {unknownShots} unresolved</small></article>
      </div>

      {teamRows.length ? (
        <div className={styles.teamSection}>
          <div className={styles.sectionHeading}>
            <strong>Team-level picture</strong>
            <span>Only attribution produced by the canonical event layer is shown.</span>
          </div>
          <div className={styles.teamGrid}>
            {teamRows.map(({ teamId, stats }) => {
              const teamShots = record(stats.shots);
              return (
                <article className={styles.teamCard} key={teamId}>
                  <div className={styles.teamName}>{humanize(teamId)}</div>
                  <div className={styles.teamStats}>
                    <div><span>Possession share</span><strong>{percentText(stats.possession_share_rate)}</strong></div>
                    <div><span>Passes</span><strong>{count(stats.passes)}</strong></div>
                    <div><span>Turnovers</span><strong>{count(stats.turnovers)}</strong></div>
                    <div><span>Rebounds</span><strong>{count(stats.rebounds)}</strong></div>
                    <div><span>Shots</span><strong>{count(teamShots.attempts)}</strong></div>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      ) : null}

      <div className={styles.evidenceStrip} aria-label="Story evidence quality">
        <div><span>Actor attribution</span><strong>{actorAttribution === null ? "—" : `${actorAttribution.toFixed(1)}%`}</strong></div>
        <div><span>Team attribution</span><strong>{teamAttribution === null ? "—" : `${teamAttribution.toFixed(1)}%`}</strong></div>
        <div><span>Possession confidence</span><strong>{possessionConfidence === null ? "—" : `${possessionConfidence.toFixed(1)}%`}</strong></div>
        <div><span>Unresolved shot outcomes</span><strong>{unknownShotRate === null ? "—" : `${unknownShotRate.toFixed(1)}%`}</strong></div>
      </div>

      <p className={styles.boundaryNote}>
        Counts above are measured P2 evidence from this run, not a manually entered box score. Candidate events and unresolved outcomes stay labelled as such; tactical conclusions remain outside this beta summary.
      </p>
    </section>
  );
}
