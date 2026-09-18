export type P3DerivationRow = {
  id: string;
  derivation_version: string;
  configuration_version: string;
  source_versions: Record<string, unknown> | null;
  configuration: Record<string, unknown> | null;
  output_summary: Record<string, unknown> | null;
  generated_at: string;
};

export type P3ReviewItemRow = {
  id: string;
  clip_id: string | null;
  review_type: string;
  source_event_ids: string[];
  source_possession_id: string | null;
  game_player_ids: string[];
  team_id: string | null;
  confidence: number | null;
  support_state: string;
  completeness: string;
  reason_shown: Record<string, unknown> | null;
  created_at: string;
};

export type P3ClipRow = {
  id: string;
  artifact_id: string | null;
  review_type: string;
  source_start_ms: number;
  source_end_ms: number;
  clip_start_ms: number;
  clip_end_ms: number;
  context_policy_version: string;
  generation_version: string;
  status: string;
  error: string | null;
};

export type P3ArtifactRow = {
  id: string;
  kind: string;
  bucket: string | null;
  object_path: string;
  content_type: string | null;
  size_bytes: number | null;
  checksum: string | null;
  metadata: Record<string, unknown> | null;
};

export type P3SourceEventRow = {
  source_event_id: string;
  start_ms: number | null;
  end_ms: number | null;
  frame_start: number;
  frame_end: number;
};

export type P3ReviewRow = {
  review_item_id: string;
  review_state: string;
  corrected_event_type: string | null;
  corrected_game_player_id: string | null;
  note: string | null;
  updated_at: string;
};

type AssembleInput = {
  analysisRunId: string;
  gameId: string;
  derivation: P3DerivationRow;
  reviewItems: P3ReviewItemRow[];
  clips: P3ClipRow[];
  artifacts: P3ArtifactRow[];
  sourceEvents: P3SourceEventRow[];
  myReviews: P3ReviewRow[];
  signedUrlsByArtifactId: Record<string, string | null>;
};

function countBy<T>(rows: T[], key: (row: T) => string) {
  const counts: Record<string, number> = {};
  for (const row of rows) {
    const value = key(row);
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

export function assembleThingsToReview(input: AssembleInput) {
  const clipById = new Map(input.clips.map((row) => [row.id, row]));
  const artifactById = new Map(input.artifacts.map((row) => [row.id, row]));
  const eventById = new Map(input.sourceEvents.map((row) => [row.source_event_id, row]));
  const reviewByItemId = new Map(input.myReviews.map((row) => [row.review_item_id, row]));

  const items = input.reviewItems.map((item) => {
    const sourceEvents = item.source_event_ids
      .map((eventId) => eventById.get(eventId))
      .filter((row): row is P3SourceEventRow => Boolean(row));
    const startValues = sourceEvents
      .map((row) => row.start_ms)
      .filter((value): value is number => typeof value === "number");
    const endValues = sourceEvents
      .map((row) => row.end_ms)
      .filter((value): value is number => typeof value === "number");
    const frameStarts = sourceEvents.map((row) => row.frame_start);
    const frameEnds = sourceEvents.map((row) => row.frame_end);

    const clip = item.clip_id ? clipById.get(item.clip_id) ?? null : null;
    const artifact = clip?.artifact_id
      ? artifactById.get(clip.artifact_id) ?? null
      : null;
    const myReview = reviewByItemId.get(item.id) ?? null;

    return {
      id: item.id,
      review_type: item.review_type,
      source_event_ids: item.source_event_ids,
      source_possession_id: item.source_possession_id,
      game_player_ids: item.game_player_ids,
      team_id: item.team_id,
      confidence: item.confidence,
      support_state: item.support_state,
      completeness: item.completeness,
      reason_shown: item.reason_shown ?? {},
      source_time: {
        start_ms: startValues.length ? Math.min(...startValues) : null,
        end_ms: endValues.length ? Math.max(...endValues) : null,
        frame_start: frameStarts.length ? Math.min(...frameStarts) : null,
        frame_end: frameEnds.length ? Math.max(...frameEnds) : null,
      },
      clip: clip
        ? {
            id: clip.id,
            status: clip.status,
            error: clip.error,
            source_start_ms: clip.source_start_ms,
            source_end_ms: clip.source_end_ms,
            clip_start_ms: clip.clip_start_ms,
            clip_end_ms: clip.clip_end_ms,
            context_policy_version: clip.context_policy_version,
            generation_version: clip.generation_version,
            artifact: artifact
              ? {
                  id: artifact.id,
                  content_type: artifact.content_type,
                  size_bytes: artifact.size_bytes,
                  checksum: artifact.checksum,
                  playable_url: input.signedUrlsByArtifactId[artifact.id] ?? null,
                }
              : null,
          }
        : null,
      my_review: myReview
        ? {
            state: myReview.review_state,
            corrected_event_type: myReview.corrected_event_type,
            corrected_game_player_id: myReview.corrected_game_player_id,
            note: myReview.note,
            updated_at: myReview.updated_at,
          }
        : null,
      created_at: item.created_at,
    };
  });

  items.sort((left, right) => {
    const leftStart = left.source_time.start_ms;
    const rightStart = right.source_time.start_ms;
    if (leftStart !== null && rightStart !== null && leftStart !== rightStart) {
      return leftStart - rightStart;
    }
    if (leftStart !== null && rightStart === null) return -1;
    if (leftStart === null && rightStart !== null) return 1;
    const created = left.created_at.localeCompare(right.created_at);
    return created !== 0 ? created : left.id.localeCompare(right.id);
  });

  const clipStatuses = input.clips.map((row) => row.status);
  return {
    analysis_run_id: input.analysisRunId,
    game_id: input.gameId,
    derivation: {
      id: input.derivation.id,
      derivation_version: input.derivation.derivation_version,
      configuration_version: input.derivation.configuration_version,
      source_versions: input.derivation.source_versions ?? {},
      configuration: input.derivation.configuration ?? {},
      generated_at: input.derivation.generated_at,
    },
    ordering: "source_time_asc",
    ranking_applied: false,
    summary: {
      total_items: items.length,
      counts_by_type: countBy(items, (row) => row.review_type),
      counts_by_support_state: countBy(items, (row) => row.support_state),
      clips: {
        planned: clipStatuses.filter((value) => value === "planned").length,
        generating: clipStatuses.filter((value) => value === "generating").length,
        ready: clipStatuses.filter((value) => value === "ready").length,
        failed: clipStatuses.filter((value) => value === "failed").length,
      },
      reviewed_by_me: items.filter((row) => row.my_review !== null).length,
      derivation_output: input.derivation.output_summary ?? {},
    },
    items,
  };
}