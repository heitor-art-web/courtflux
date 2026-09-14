import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import { workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

function workerMessage(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const data = value as Record<string, unknown>;
  if (typeof data.detail === "string" && data.detail.trim()) return data.detail.trim();
  if (typeof data.error === "string" && data.error.trim()) return data.error.trim();
  return null;
}

export async function POST(
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
      .select("workspace_id, role")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();

    if (membershipError) throw membershipError;
    if (!membership?.workspace_id) {
      return Response.json({ error: "No active CourtFlux workspace" }, { status: 403 });
    }
    if (!["owner", "admin", "analyst"].includes(membership.role ?? "")) {
      return Response.json({ error: "Your workspace role cannot stop analyses" }, { status: 403 });
    }

    const { id } = await context.params;
    const workspaceId = membership.workspace_id as string;
    const admin = createAdminClient();

    const { data: run, error: runError } = await admin
      .from("analysis_runs")
      .select("id, status")
      .eq("id", id)
      .eq("workspace_id", workspaceId)
      .maybeSingle();

    if (runError) throw runError;
    if (!run) return Response.json({ error: "Analysis not found" }, { status: 404 });

    if (run.status === "canceled") {
      return Response.json({ status: "canceled", already_terminal: true });
    }
    if (run.status === "completed" || run.status === "failed") {
      return Response.json(
        { error: `Analysis is already ${run.status}`, status: run.status },
        { status: 409 },
      );
    }
    if (run.status !== "queued" && run.status !== "running") {
      return Response.json(
        { error: `Analysis cannot be stopped from status ${run.status}` },
        { status: 409 },
      );
    }

    const { data: job, error: jobError } = await admin
      .from("analysis_jobs")
      .select("id, status")
      .eq("run_id", run.id)
      .eq("workspace_id", workspaceId)
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();

    if (jobError) throw jobError;
    if (!job?.id) {
      return Response.json({ error: "No worker job is attached to this analysis" }, { status: 409 });
    }

    const upstream = await workerFetch(`/jobs/${encodeURIComponent(job.id)}/cancel`, {
      method: "POST",
      signal: AbortSignal.timeout(15_000),
    });
    const text = await upstream.text();
    let payload: Record<string, unknown> = {};
    try {
      payload = text ? JSON.parse(text) as Record<string, unknown> : {};
    } catch {
      payload = { detail: text };
    }

    if (!upstream.ok) {
      return Response.json(
        {
          error: workerMessage(payload) || `Could not stop analysis (worker HTTP ${upstream.status})`,
          worker_status: upstream.status,
        },
        { status: upstream.status === 409 ? 409 : 503 },
      );
    }

    const workerStatus = typeof payload.status === "string" ? payload.status : null;
    if (workerStatus === "canceled") {
      const completedAt = new Date().toISOString();
      await Promise.all([
        admin
          .from("analysis_jobs")
          .update({ status: "canceled", completed_at: completedAt, error: null })
          .eq("id", job.id)
          .eq("workspace_id", workspaceId)
          .in("status", ["queued", "running"]),
        admin
          .from("analysis_runs")
          .update({ status: "canceled", completed_at: completedAt })
          .eq("id", run.id)
          .eq("workspace_id", workspaceId)
          .in("status", ["queued", "running"]),
      ]);
      return Response.json({ status: "canceled", job_id: job.id, run_id: run.id });
    }

    if (workerStatus === "completed" || workerStatus === "failed") {
      return Response.json(
        { error: `Analysis became ${workerStatus} before it could be stopped`, status: workerStatus },
        { status: 409 },
      );
    }

    return Response.json(
      { status: "cancel_requested", job_id: job.id, run_id: run.id },
      { status: 202 },
    );
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
