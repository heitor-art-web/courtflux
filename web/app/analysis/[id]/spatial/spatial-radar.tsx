"use client";

import { useEffect, useMemo, useState } from "react";
import type { CourtPoint, SpatialViewerFrame } from "@/lib/p4/spatial-viewer";
import styles from "./spatial-radar.module.css";

const COURT_LENGTH_M = 28;
const COURT_WIDTH_M = 15;
const BASKET_FROM_ENDLINE_M = 1.575;
const TRAIL_FRAMES = 24;

export type SpatialRunMeta = {
  id: string;
  status: string;
  contractVersion: string;
  spatialRuntimeVersion: string;
  createdAt: string | null;
  completedAt: string | null;
};

type Props = {
  run: SpatialRunMeta | null;
  frames: SpatialViewerFrame[];
  rejectedRows: number;
  dataError: string | null;
};

function formatSeconds(timestampMs: number) {
  return `${(timestampMs / 1000).toFixed(2)} s`;
}

function formatNumber(value: number | null, digits = 2) {
  return value === null ? "—" : value.toFixed(digits);
}

function classForState(state: string) {
  if (state === "invalid") return styles.invalid;
  if (state === "degraded" || state.includes("aging")) return styles.degraded;
  return styles.good;
}

function teamClass(index: number) {
  const classes = [styles.team0, styles.team1, styles.team2, styles.team3];
  return classes[index % classes.length];
}

function pathFromPoints(points: CourtPoint[]) {
  return points.map(([x, y], pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${x} ${y}`).join(" ");
}

export function SpatialRadar({ run, frames, rejectedRows, dataError }: Props) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showRaw, setShowRaw] = useState(true);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [showTrails, setShowTrails] = useState(true);
  const [showVelocity, setShowVelocity] = useState(true);
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null);

  const current = frames[index] ?? null;

  useEffect(() => {
    if (!playing || !current || index >= frames.length - 1) return;
    const next = frames[index + 1];
    const sourceDelay = Math.max(1, next.timestampMs - current.timestampMs);
    const delay = Math.min(500, Math.max(40, sourceDelay / playbackRate));
    const timer = window.setTimeout(() => {
      setIndex((value) => {
        const nextIndex = Math.min(value + 1, frames.length - 1);
        if (nextIndex >= frames.length - 1) setPlaying(false);
        return nextIndex;
      });
    }, delay);
    return () => window.clearTimeout(timer);
  }, [current, frames, index, playbackRate, playing]);

  const teamOrder = useMemo(() => {
    const ids = new Set<string>();
    for (const frame of frames) {
      for (const player of frame.players) {
        if (player.localTeamId) ids.add(player.localTeamId);
      }
    }
    return [...ids].sort();
  }, [frames]);

  const trails = useMemo(() => {
    if (!current || !showTrails) return new Map<number, CourtPoint[]>();
    const start = Math.max(0, index - TRAIL_FRAMES + 1);
    const result = new Map<number, CourtPoint[]>();
    for (const frame of frames.slice(start, index + 1)) {
      for (const player of frame.players) {
        const courtPoint = player.smoothedPositionM ?? player.rawPositionM;
        if (!courtPoint) continue;
        const existing = result.get(player.trackId) ?? [];
        existing.push(courtPoint);
        result.set(player.trackId, existing);
      }
    }
    return result;
  }, [current, frames, index, showTrails]);

  const selected = current?.players.find((player) => player.trackId === selectedTrackId) ?? null;

  if (dataError) {
    return (
      <section className={styles.emptyState}>
        <p className={styles.eyebrow}>P4 spatial diagnostics</p>
        <h1>Spatial data unavailable</h1>
        <p>{dataError}</p>
        <p className={styles.muted}>
          The viewer is fail-closed and does not synthesize spatial state when the P4 persistence layer is unavailable.
        </p>
      </section>
    );
  }

  if (!run || frames.length === 0 || !current) {
    return (
      <section className={styles.emptyState}>
        <p className={styles.eyebrow}>P4 spatial diagnostics</p>
        <h1>No materialized snapshots</h1>
        <p>This analysis does not currently have queryable <code>p4.spatial_snapshot.v1</code> frames.</p>
      </section>
    );
  }

  return (
    <section className={styles.workspace}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Internal diagnostic foundation</p>
          <h1>Spatial Radar / Court Viewer</h1>
          <p className={styles.subtitle}>
            P1/P4-only geometry and motion evidence. No possession, offense/defense, ball-handler or tactical labels.
          </p>
        </div>
        <div className={styles.runMeta}>
          <span>Spatial run</span>
          <strong>{run.id.slice(0, 8)}</strong>
          <span className={styles.muted}>{run.status} · {run.spatialRuntimeVersion}</span>
        </div>
      </header>

      <div className={styles.statusStrip}>
        <div><span>Frame</span><strong>{current.frameId}</strong></div>
        <div><span>Timestamp</span><strong>{formatSeconds(current.timestampMs)}</strong></div>
        <div><span>Geometry</span><strong className={classForState(current.geometryQuality.state)}>{current.geometryQuality.state}</strong></div>
        <div><span>Eligibility</span><strong className={classForState(current.eligibility.state)}>{current.eligibility.state}</strong></div>
        <div><span>Metric coverage</span><strong>{Math.round(current.eligibility.metricPlayerFraction * 100)}%</strong></div>
      </div>

      <div className={styles.mainGrid}>
        <div className={styles.courtPanel}>
          <div className={styles.courtToolbar}>
            <label><input type="checkbox" checked={showRaw} onChange={(event) => setShowRaw(event.target.checked)} /> Raw</label>
            <label><input type="checkbox" checked={showSmoothed} onChange={(event) => setShowSmoothed(event.target.checked)} /> Smoothed</label>
            <label><input type="checkbox" checked={showTrails} onChange={(event) => setShowTrails(event.target.checked)} /> Trails</label>
            <label><input type="checkbox" checked={showVelocity} onChange={(event) => setShowVelocity(event.target.checked)} /> Velocity</label>
          </div>

          <div className={styles.courtWrap}>
            <svg
              className={styles.court}
              viewBox={`-0.5 -0.5 ${COURT_LENGTH_M + 1} ${COURT_WIDTH_M + 1}`}
              role="img"
              aria-label={`Spatial court at frame ${current.frameId}`}
            >
              <rect className={styles.courtFloor} x="0" y="0" width={COURT_LENGTH_M} height={COURT_WIDTH_M} rx="0.12" />
              <rect className={styles.courtLine} x="0" y="0" width={COURT_LENGTH_M} height={COURT_WIDTH_M} />
              <line className={styles.courtLine} x1="14" y1="0" x2="14" y2={COURT_WIDTH_M} />
              <circle className={styles.courtLine} cx="14" cy="7.5" r="1.8" />
              <rect className={styles.courtLine} x="0" y="5.05" width="5.8" height="4.9" />
              <rect className={styles.courtLine} x="22.2" y="5.05" width="5.8" height="4.9" />
              <circle className={styles.basket} cx={BASKET_FROM_ENDLINE_M} cy="7.5" r="0.23" />
              <circle className={styles.basket} cx={COURT_LENGTH_M - BASKET_FROM_ENDLINE_M} cy="7.5" r="0.23" />

              {[...trails.entries()].map(([trackId, points]) =>
                points.length > 1 ? <path key={`trail-${trackId}`} className={styles.trail} d={pathFromPoints(points)} /> : null,
              )}

              {current.players.map((player) => {
                const teamIndex = player.localTeamId ? Math.max(0, teamOrder.indexOf(player.localTeamId)) : 3;
                const displayPoint = player.smoothedPositionM ?? player.rawPositionM;
                const selectTrack = () => setSelectedTrackId(player.trackId);
                return (
                  <g key={player.trackId} className={selectedTrackId === player.trackId ? styles.selectedPlayer : undefined}>
                    {showRaw && player.rawPositionM ? (
                      <circle className={styles.rawPoint} cx={player.rawPositionM[0]} cy={player.rawPositionM[1]} r="0.22" />
                    ) : null}
                    {showVelocity && displayPoint && player.velocityMps ? (
                      <line
                        className={styles.velocity}
                        x1={displayPoint[0]}
                        y1={displayPoint[1]}
                        x2={displayPoint[0] + player.velocityMps[0] * 0.35}
                        y2={displayPoint[1] + player.velocityMps[1] * 0.35}
                      />
                    ) : null}
                    {showSmoothed && player.smoothedPositionM ? (
                      <g
                        className={styles.svgButton}
                        role="button"
                        tabIndex={0}
                        aria-label={`Select track ${player.trackId}`}
                        onClick={selectTrack}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            selectTrack();
                          }
                        }}
                      >
                        <circle className={`${styles.playerPoint} ${teamClass(teamIndex)}`} cx={player.smoothedPositionM[0]} cy={player.smoothedPositionM[1]} r="0.34" />
                        <text className={styles.trackLabel} x={player.smoothedPositionM[0]} y={player.smoothedPositionM[1] + 0.11}>{player.trackId}</text>
                      </g>
                    ) : null}
                  </g>
                );
              })}
            </svg>
          </div>

          <div className={styles.timeline}>
            <button type="button" onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button>
            <button type="button" onClick={() => { setPlaying(false); setIndex(0); }}>Start</button>
            <input
              aria-label="Spatial snapshot timeline"
              type="range"
              min={0}
              max={frames.length - 1}
              value={index}
              onChange={(event) => { setPlaying(false); setIndex(Number(event.target.value)); }}
            />
            <select aria-label="Playback speed" value={playbackRate} onChange={(event) => setPlaybackRate(Number(event.target.value))}>
              <option value={0.5}>0.5×</option>
              <option value={1}>1×</option>
              <option value={2}>2×</option>
              <option value={4}>4×</option>
            </select>
            <span>{index + 1} / {frames.length}</span>
          </div>
        </div>

        <aside className={styles.sidePanel}>
          <section className={styles.card}>
            <h2>Geometry diagnostics</h2>
            <dl>
              <div><dt>Source</dt><dd>{current.geometry.homographySource ?? "—"}</dd></div>
              <div><dt>Reason</dt><dd>{current.geometryQuality.reason}</dd></div>
              <div><dt>Age</dt><dd>{current.geometryQuality.homographyAgeFrames ?? "—"} frames</dd></div>
              <div><dt>Anchors / inliers</dt><dd>{current.geometryQuality.anchorCount ?? "—"} / {current.geometryQuality.inlierCount ?? "—"}</dd></div>
              <div><dt>RMSE</dt><dd>{formatNumber(current.geometryQuality.reprojectionRmsePx)} px</dd></div>
              <div><dt>Projection uncertainty</dt><dd>{formatNumber(current.geometryQuality.projectionUncertaintyM)} m</dd></div>
            </dl>
          </section>

          <section className={styles.card}>
            <h2>Eligibility</h2>
            <p className={`${styles.statePill} ${classForState(current.eligibility.state)}`}>{current.eligibility.state}</p>
            <ul className={styles.reasonList}>{current.eligibility.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
            <p className={styles.muted}>{current.eligibility.metricPlayerCount}/{current.eligibility.visiblePlayerCount} visible players have metric positions.</p>
          </section>

          <section className={styles.card}>
            <h2>Selected track</h2>
            {selected ? (
              <dl>
                <div><dt>Track ID</dt><dd>{selected.trackId}</dd></div>
                <div><dt>Local team</dt><dd>{selected.localTeamId ?? "unknown"}</dd></div>
                <div><dt>Raw XY</dt><dd>{selected.rawPositionM ? `${selected.rawPositionM[0].toFixed(2)}, ${selected.rawPositionM[1].toFixed(2)}` : "—"}</dd></div>
                <div><dt>Smoothed XY</dt><dd>{selected.smoothedPositionM ? `${selected.smoothedPositionM[0].toFixed(2)}, ${selected.smoothedPositionM[1].toFixed(2)}` : "—"}</dd></div>
                <div><dt>Speed</dt><dd>{formatNumber(selected.speedMps)} m/s</dd></div>
                <div><dt>Trajectory</dt><dd>{selected.trajectoryState ?? "—"}</dd></div>
                <div><dt>Kinematics</dt><dd>{selected.kinematicsState ?? "—"}</dd></div>
              </dl>
            ) : <p className={styles.muted}>Select a smoothed player marker to inspect its spatial state.</p>}
          </section>

          {rejectedRows > 0 ? <p className={styles.warning}>{rejectedRows} persisted snapshot row(s) were rejected by the viewer parser.</p> : null}
        </aside>
      </div>
    </section>
  );
}
