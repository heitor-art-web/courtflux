import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import { workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

function parseSupabaseSource(value: string | null) {
  if (!value?.startsWith("supabase://")) return null;
  const remainder = value.slice("supabase://".length);
  const slash = remainder.indexOf("/");
  if (slash <= 0 || slash === remainder.length - 1) return null;
  return {
    bucket: remainder.slice(0, slash),
    path: remainder.slice(slash + 1),
  };
}

function sourceKind(value: string | null) {
  if (!value) return "unknown";
  if (value.startsWith("supabase://")) return "upload";
  if (/youtu(?:\.be|be\.com)/i.test(value)) return "youtube";
  return "other";
}

function publicWorkerState(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const state = value as Record<string, unknown>;
  const progress = state.progress && typeof state.progress === "object"
    ? state.progress as Record<string, unknown>
    : null;
  const artifacts = state.artifacts && typeof state.artifacts === "object"
    ? state.artifacts as Record<string, unknown>
    : null;

  return {
    status: typeof state.status === "string" ? state.status : null,
    started_at: typeof state.started_at === "string" ? state.started_at : null,
    completed_at: typeof state.completed_at === "string" ? state.completed_at : null,
    progress: progress
      ? {
          phase: typeof progress.phase === "string" ? progress.phase : null,
          percent: typeof progress.percent === "number" ? progress.percent : null,
          processed_seconds: typeof progress.processed_seconds === "number" ? progress.processed_seconds : null,
          total_seconds: typeof progress.total_seconds === "number" ? progress.total_seconds : null,
          frames_processed: typeof progress.frames_processed === "number" ? progress.frames_processed : null,
          elapsed_sec: typeof progress.elapsed_sec === "number" ? progress.elapsed_sec : null,
          eta_sec: typeof progress.eta_sec === "number" ? progress.eta_sec : null,
          estimated_completion_at: typeof progress.estimated_completion_at === "string" ? progress.estimated_completion_at : null,
          last_progress_at: typeof progress.last_progress_at === "string" ? progress.last_progress_at : null,
        }
      : null,
    artifacts: artifacts
      ? {
          preview: artifacts.preview === true,
          analysis_video: artifacts.analysis_video === true,
        }
      : null,
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
    const admin = createAdminClient();
    const workspaceId = membership.workspace_id as string;

    const { data: run, error: runError } = await admin
      .from("analysis_runs")
      .select(
        "id, workspace_id, game_id, status, video_source, start_time_sec, end_time_sec, parameters, summary, started_at, completed_at, created_at, updated_at, games(id, title, competition, game_date, source_type, metadata)",
      )
      .eq("id", id)
      .eq("workspace_id", workspaceId)
      .maybeSingle();

    if (runError) throw runError;
    if (!run) return Response.json({ error: "Analysis not found" }, { status: 404 });

    const { data: job, error: jobError } = await admin
      .from("analysis_jobs")
      .select("id, status, error, log_tail, queued_at, started_at, completed_at, created_at, updated_at")
      .eq("run_id", run.id)
      .eq("workspace_id", workspaceId)
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    if (jobError) throw jobError;

    let worker = null;
    if (job?.id) {
      try {
        const response = await workerFetch(`/jobs/${encodeURIComponent(job.id)}`, {
          signal: AbortSignal.timeout(2500),
        });
        if (response.ok) {
          worker = publicWorkerState(await response.json());
        }
      } catch {
        // The durable Supabase state remains authoritative if the worker is temporarily unreachable.
      }
    }

    const game = Array.isArray(run.games) ? run.games[0] : run.games;
    const kind = sourceKind(run.video_source);
    let playableUrl: string | null = kind === "youtube" ? run.video_source : null;
    let retention: { expires_at: string | null; deleted_at: string | null } | null = null;

    const privateSource = parseSupabaseSource(run.video_source);
    if (privateSource?.bucket === "analysis-videos") {
      const expectedPrefix = `${workspaceId}/`;
      if (privateSource.path.startsWith(expectedPrefix)) {
        const { data: retentionRow } = await admin
          .from("analysis_video_uploads")
          .select("expires_at, deleted_at")
          .eq("workspace_id", workspaceId)
          .eq("object_path", privateSource.path)
          .maybeSingle();

        retention = retentionRow
          ? { expires_at: retentionRow.expires_at, deleted_at: retentionRow.deleted_at }
          : null;

        if (!retentionRow?.deleted_at) {
          const { data: signed } = await admin.storage
            .from(privateSource.bucket)
            .createSignedUrl(privateSource.path, 3600);
          playableUrl = signed?.signedUrl ?? null;
        }
      }
    }

    return Response.json({
      run: {
        id: run.id,
        status: run.status,
        start_time_sec: run.start_time_sec,
        end_time_sec: run.end_time_sec,
        parameters: run.parameters,
        summary: run.summary,
        started_at: run.started_at,
        completed_at: run.completed_at,
        created_at: run.created_at,
        updated_at: run.updated_at,
      },
      job: job
        ? {
            id: job.id,
            status: job.status,
            error: job.error,
            log_tail: job.log_tail,
            queued_at: job.queued_at,
            started_at: job.started_at,
            completed_at: job.completed_at,
            created_at: job.created_at,
            updated_at: job.updated_at,
          }
        : null,
      worker,
      game: game
        ? {
            id: game.id,
            title: game.title,
            competition: game.competition,
            game_date: game.game_date,
            source_type: game.source_type,
            metadata: game.metadata,
          }
        : null,
      source: {
        kind,
        playable_url: playableUrl,
        retention,
      },
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
