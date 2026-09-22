export type CourtPoint = readonly [number, number];

export type SpatialViewerPlayer = {
  trackId: number;
  localTeamId: string | null;
  visible: boolean;
  rawPositionM: CourtPoint | null;
  smoothedPositionM: CourtPoint | null;
  velocityMps: CourtPoint | null;
  speedMps: number | null;
  trajectoryState: string | null;
  kinematicsState: string | null;
  observationConfidence: number | null;
  positionConfidence: number | null;
};

export type SpatialViewerFrame = {
  frameId: number;
  timestampMs: number;
  contractVersion: string;
  sourceRuntimeVersion: string | null;
  geometry: {
    valid: boolean;
    courtVisible: boolean;
    homographySource: string | null;
    reprojectionRmsePx: number | null;
  };
  geometryQuality: {
    state: string;
    reason: string;
    homographyAgeFrames: number | null;
    anchorCount: number | null;
    inlierCount: number | null;
    reprojectionRmsePx: number | null;
    projectionUncertaintyM: number | null;
  };
  eligibility: {
    state: string;
    reasons: string[];
    visiblePlayerCount: number;
    metricPlayerCount: number;
    metricPlayerFraction: number;
  };
  players: SpatialViewerPlayer[];
};

export type SpatialSnapshotRow = {
  payload: unknown;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nonNegativeInteger(value: unknown): number | null {
  return Number.isInteger(value) && typeof value === "number" && value >= 0 ? value : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function point(value: unknown): CourtPoint | null {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const x = finiteNumber(value[0]);
  const y = finiteNumber(value[1]);
  return x === null || y === null ? null : [x, y];
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.length > 0);
}

function parsePlayer(value: unknown): SpatialViewerPlayer | null {
  if (!isRecord(value)) return null;
  const trackId = nonNegativeInteger(value.track_id);
  if (trackId === null || typeof value.visible !== "boolean") return null;

  return {
    trackId,
    localTeamId: value.local_team_id === null ? null : text(value.local_team_id),
    visible: value.visible,
    rawPositionM: point(value.court_position_m),
    smoothedPositionM: point(value.smoothed_position_m),
    velocityMps: point(value.velocity_mps),
    speedMps: finiteNumber(value.speed_mps),
    trajectoryState: value.trajectory_state === null ? null : text(value.trajectory_state),
    kinematicsState: value.kinematics_state === null ? null : text(value.kinematics_state),
    observationConfidence: finiteNumber(value.observation_confidence),
    positionConfidence: finiteNumber(value.position_confidence),
  };
}

export function parseSpatialSnapshotPayload(payload: unknown): SpatialViewerFrame | null {
  if (!isRecord(payload) || payload.contract_version !== "p4.spatial_snapshot.v1") return null;

  const frameId = nonNegativeInteger(payload.frame_id);
  const timestampMs = nonNegativeInteger(payload.timestamp_ms);
  const geometry = isRecord(payload.geometry) ? payload.geometry : null;
  const quality = isRecord(payload.geometry_quality) ? payload.geometry_quality : null;
  const eligibility = isRecord(payload.spatial_eligibility) ? payload.spatial_eligibility : null;
  if (frameId === null || timestampMs === null || !geometry || !quality || !eligibility) return null;
  if (typeof geometry.geometry_valid !== "boolean" || typeof geometry.court_visible !== "boolean") return null;

  const qualityState = text(quality.state);
  const qualityReason = text(quality.reason);
  const eligibilityState = text(eligibility.state);
  const visiblePlayerCount = nonNegativeInteger(eligibility.visible_player_count);
  const metricPlayerCount = nonNegativeInteger(eligibility.metric_player_count);
  const metricPlayerFraction = finiteNumber(eligibility.metric_player_fraction);
  if (
    !qualityState ||
    !qualityReason ||
    !eligibilityState ||
    visiblePlayerCount === null ||
    metricPlayerCount === null ||
    metricPlayerFraction === null
  ) {
    return null;
  }

  const players = Array.isArray(payload.players)
    ? payload.players.map(parsePlayer).filter((item): item is SpatialViewerPlayer => item !== null)
    : [];
  if (!Array.isArray(payload.players) || players.length !== payload.players.length) return null;

  return {
    frameId,
    timestampMs,
    contractVersion: "p4.spatial_snapshot.v1",
    sourceRuntimeVersion: payload.source_runtime_version === null ? null : text(payload.source_runtime_version),
    geometry: {
      valid: geometry.geometry_valid,
      courtVisible: geometry.court_visible,
      homographySource: geometry.homography_source === null ? null : text(geometry.homography_source),
      reprojectionRmsePx: finiteNumber(geometry.reprojection_rmse_px),
    },
    geometryQuality: {
      state: qualityState,
      reason: qualityReason,
      homographyAgeFrames: nonNegativeInteger(quality.homography_age_frames),
      anchorCount: nonNegativeInteger(quality.anchor_count),
      inlierCount: nonNegativeInteger(quality.inlier_count),
      reprojectionRmsePx: finiteNumber(quality.reprojection_rmse_px),
      projectionUncertaintyM: finiteNumber(quality.projection_uncertainty_m),
    },
    eligibility: {
      state: eligibilityState,
      reasons: stringArray(eligibility.reasons),
      visiblePlayerCount,
      metricPlayerCount,
      metricPlayerFraction,
    },
    players,
  };
}

export function parseSpatialSnapshotRows(rows: SpatialSnapshotRow[]) {
  const frames: SpatialViewerFrame[] = [];
  let rejected = 0;
  for (const row of rows) {
    const parsed = parseSpatialSnapshotPayload(row.payload);
    if (parsed) frames.push(parsed);
    else rejected += 1;
  }
  frames.sort((a, b) => a.timestampMs - b.timestampMs || a.frameId - b.frameId);
  return { frames, rejected };
}
