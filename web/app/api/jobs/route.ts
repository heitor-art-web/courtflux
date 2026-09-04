import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import { workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let runId: string | null = null;
  let jobId: string | null = null;

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

    const payload = await request.json();
    const workspaceId = membership.workspace_id as string;
    runId = crypto.randomUUID();
    jobId = crypto.randomUUID();
    const admin = createAdminClient();

    const { data: entitlement } = await admin
      .from("workspace_entitlements")
      .select("plan_code, status, features, limits")
      .eq("workspace_id", workspaceId)
      .maybeSingle();

    const { error: runError } = await admin.from("analysis_runs").insert({
      id: runId,
      workspace_id: workspaceId,
      created_by: user.id,
      status: "queued",
      video_source: typeof payload.video_source === "string" ? payload.video_source : null,
      start_time_sec: Number(payload.start_time_sec ?? 0),
      end_time_sec: payload.end_time_sec == null ? null : Number(payload.end_time_sec),
      parameters: payload,
    });
    if (runError) throw runError;

    const { error: jobError } = await admin.from("analysis_jobs").insert({
      id: jobId,
      run_id: runId,
      workspace_id: workspaceId,
      status: "queued",
      request: payload,
    });
    if (jobError) throw jobError;

    const upstream = await workerFetch("/jobs", {
      method: "POST",
      body: JSON.stringify({
        ...payload,
        platform_run_id: runId,
        platform_job_id: jobId,
        workspace_id: workspaceId,
        user_id: user.id,
        subscription_plan: entitlement?.plan_code ?? "free",
      }),
    });

    const text = await upstream.text();
    let data: Record<string, unknown> = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }

    if (!upstream.ok) {
      const errorText = String(data.detail ?? data.error ?? `Worker HTTP ${upstream.status}`);
      await Promise.all([
        admin.from("analysis_jobs").update({ status: "failed", error: errorText, completed_at: new Date().toISOString() }).eq("id", jobId),
        admin.from("analysis_runs").update({ status: "failed", completed_at: new Date().toISOString() }).eq("id", runId),
      ]);
      return Response.json({ ...data, run_id: runId, job_id: jobId }, { status: upstream.status });
    }

    return Response.json({ ...data, run_id: runId, job_id: jobId });
  } catch (error) {
    const errorText = error instanceof Error ? error.message : String(error);
    try {
      if (runId || jobId) {
        const admin = createAdminClient();
        const completedAt = new Date().toISOString();
        const writes = [];
        if (jobId) writes.push(admin.from("analysis_jobs").update({ status: "failed", error: errorText, completed_at: completedAt }).eq("id", jobId));
        if (runId) writes.push(admin.from("analysis_runs").update({ status: "failed", completed_at: completedAt }).eq("id", runId));
        await Promise.all(writes);
      }
    } catch {
      // Preserve the original request error.
    }
    return Response.json({ error: errorText }, { status: 503 });
  }
}
