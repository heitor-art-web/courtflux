import { assembleThingsToReview } from "@/lib/p3/things-to-review";
import type {
  P3ArtifactRow,
  P3ClipRow,
  P3DerivationRow,
  P3ReviewItemRow,
  P3ReviewRow,
  P3SourceEventRow,
} from "@/lib/p3/things-to-review";
import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

async function fetchPaged<T>(
  fetchPage: (from: number, to: number) => PromiseLike<{
    data: T[] | null;
    error: unknown;
  }>,
) {
  const pageSize = 1000;
  const rows: T[] = [];
  for (let from = 0; from < 10_000; from += pageSize) {
    const { data, error } = await fetchPage(from, from + pageSize - 1);
    if (error) throw error;
    const page = data ?? [];
    rows.push(...page);
    if (page.length < pageSize) return rows;
  }
  throw new Error("Things to Review exceeded the 10,000-row safety limit");
}

function chunked<T>(rows: T[], size = 200) {
  const chunks: T[][] = [];
  for (let index = 0; index < rows.length; index += size) {
    chunks.push(rows.slice(index, index + size));
  }
  return chunks;
}


function emptyResponse(analysisRunId: string, gameId: string) {
  return {
    analysis_run_id: analysisRunId,
    game_id: gameId,
    derivation: null,
    ordering: "source_time_asc",
    ranking_applied: false,
    summary: {
      total_items: 0,
      counts_by_type: {},
      counts_by_support_state: {},
      clips: { planned: 0, generating: 0, ready: 0, failed: 0 },
      reviewed_by_me: 0,
      derivation_output: {},
    },
    items: [],
  };
}

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (!user) {
      return Response.json({ error: "Authentication required" }, { status: 401 });
    }

    const { data: membership, error: membershipError } = await supabase
      .from("workspace_members")
      .select("workspace_id")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();
    if (membershipError) throw membershipError;
    if (!membership?.workspace_id) {
      return Response.json({ error: "No active CourtFlux workspace" }, { status: 403 });
    }

    const { id } = await context.params;
    const workspaceId = membership.workspace_id as string;
    const admin = createAdminClient();

    const { data: run, error: runError } = await admin
      .from("analysis_runs")
      .select("id,workspace_id,game_id")
      .eq("id", id)
      .eq("workspace_id", workspaceId)
      .maybeSingle();
    if (runError) throw runError;
    if (!run) return Response.json({ error: "Analysis not found" }, { status: 404 });
    if (!run.game_id) {
      return Response.json(
        { error: "Analysis is not linked to a Game Record" },
        { status: 409 },
      );
    }

    const { data: derivation, error: derivationError } = await admin
      .from("analysis_derivations")
      .select(
        "id,derivation_version,configuration_version,source_versions,configuration,output_summary,generated_at",
      )
      .eq("workspace_id", workspaceId)
      .eq("analysis_run_id", run.id)
      .eq("game_id", run.game_id)
      .eq("layer", "p3")
      .eq("derivation_key", "things_to_review")
      .eq("status", "current")
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (derivationError) throw derivationError;
    if (!derivation) {
      return Response.json(emptyResponse(run.id, run.game_id));
    }

    const derivationId = derivation.id as string;
    const [itemRows, clipRows, eventRows] = await Promise.all([
      fetchPaged<P3ReviewItemRow>((from, to) =>
        admin
          .from("p3_review_items")
          .select(
            "id,clip_id,review_type,source_event_ids,source_possession_id,game_player_ids,team_id,confidence,support_state,completeness,reason_shown,created_at",
          )
          .eq("workspace_id", workspaceId)
          .eq("analysis_run_id", run.id)
          .eq("derivation_id", derivationId)
          .order("created_at", { ascending: true })
          .range(from, to),
      ),
      fetchPaged<P3ClipRow>((from, to) =>
        admin
          .from("p3_clips")
          .select(
            "id,artifact_id,review_type,source_start_ms,source_end_ms,clip_start_ms,clip_end_ms,context_policy_version,generation_version,status,error",
          )
          .eq("workspace_id", workspaceId)
          .eq("analysis_run_id", run.id)
          .eq("derivation_id", derivationId)
          .range(from, to),
      ),
      fetchPaged<P3SourceEventRow>((from, to) =>
        admin
          .from("p3_source_events")
          .select("source_event_id,start_ms,end_ms,frame_start,frame_end")
          .eq("workspace_id", workspaceId)
          .eq("analysis_run_id", run.id)
          .eq("derivation_id", derivationId)
          .order("frame_start", { ascending: true })
          .range(from, to),
      ),
    ]);

    const reviewItemIds = itemRows.map((row) => row.id);
    let myReviews: P3ReviewRow[] = [];
    if (reviewItemIds.length) {
      const pages = await Promise.all(
        chunked(reviewItemIds).map(async (ids) => {
          const { data, error } = await admin
            .from("p3_reviews")
            .select(
              "review_item_id,review_state,corrected_event_type,corrected_game_player_id,note,updated_at",
            )
            .eq("workspace_id", workspaceId)
            .eq("reviewer_id", user.id)
            .in("review_item_id", ids);
          if (error) throw error;
          return (data ?? []) as P3ReviewRow[];
        }),
      );
      myReviews = pages.flat();
    }

    const artifactIds = Array.from(
      new Set(
        clipRows
          .map((row) => row.artifact_id)
          .filter((value): value is string => Boolean(value)),
      ),
    );
    let artifacts: P3ArtifactRow[] = [];
    if (artifactIds.length) {
      const pages = await Promise.all(
        chunked(artifactIds).map(async (ids) => {
          const { data, error } = await admin
            .from("analysis_artifacts")
            .select(
              "id,kind,bucket,object_path,content_type,size_bytes,checksum,metadata",
            )
            .eq("run_id", run.id)
            .eq("kind", "p3_evidence_clip")
            .in("id", ids);
          if (error) throw error;
          return (data ?? []) as P3ArtifactRow[];
        }),
      );
      artifacts = pages.flat();
    }

    const signedUrlsByArtifactId: Record<string, string | null> = {};
    await Promise.all(
      artifacts.map(async (artifact) => {
        signedUrlsByArtifactId[artifact.id] = null;
        const expectedPrefix = `${workspaceId}/${run.id}/p3/clips/`;
        if (
          artifact.bucket !== "analysis-artifacts" ||
          !artifact.object_path.startsWith(expectedPrefix)
        ) {
          return;
        }
        const { data, error } = await admin.storage
          .from("analysis-artifacts")
          .createSignedUrl(artifact.object_path, 3600);
        if (!error) {
          signedUrlsByArtifactId[artifact.id] = data?.signedUrl ?? null;
        }
      }),
    );

    return Response.json(
      assembleThingsToReview({
        analysisRunId: run.id,
        gameId: run.game_id,
        derivation: derivation as P3DerivationRow,
        reviewItems: itemRows,
        clips: clipRows,
        artifacts,
        sourceEvents: eventRows,
        myReviews,
        signedUrlsByArtifactId,
      }),
      {
        headers: {
          "Cache-Control": "private, no-store",
        },
      },
    );
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}