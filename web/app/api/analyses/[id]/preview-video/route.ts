import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();
    if (!user) return Response.json({ error: "Authentication required" }, { status: 401 });

    const { data: membership, error: membershipError } = await supabase
      .from("workspace_members")
      .select("workspace_id")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();
    if (membershipError) throw membershipError;
    if (!membership?.workspace_id) return Response.json({ error: "No active CourtFlux workspace" }, { status: 403 });

    const { id } = await context.params;
    const admin = createAdminClient();
    const workspaceId = membership.workspace_id as string;

    const { data: run, error: runError } = await admin
      .from("analysis_runs")
      .select("id")
      .eq("id", id)
      .eq("workspace_id", workspaceId)
      .maybeSingle();
    if (runError) throw runError;
    if (!run) return Response.json({ error: "Analysis not found" }, { status: 404 });

    const { data: job, error: jobError } = await admin
      .from("analysis_jobs")
      .select("id")
      .eq("run_id", run.id)
      .eq("workspace_id", workspaceId)
      .order("created_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (jobError) throw jobError;
    if (!job?.id) return Response.json({ error: "Analysis worker job not found" }, { status: 404 });

    const range = request.headers.get("range");
    return proxyWorkerResponse(
      await workerFetch(`/jobs/${encodeURIComponent(job.id)}/preview-video`, {
        headers: range ? { Range: range } : undefined,
        signal: AbortSignal.timeout(10000),
      }),
    );
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
