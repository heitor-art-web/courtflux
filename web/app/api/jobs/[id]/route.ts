import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

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

    const { id } = await context.params;
    const { data, error } = await supabase
      .from("analysis_jobs")
      .select("id, status, error, run_id, analysis_runs(summary)")
      .eq("id", id)
      .maybeSingle();

    if (error) throw error;
    if (!data) return Response.json({ error: "Job not found" }, { status: 404 });

    const run = Array.isArray(data.analysis_runs) ? data.analysis_runs[0] : data.analysis_runs;
    return Response.json({
      job_id: data.id,
      run_id: data.run_id,
      status: data.status,
      error: data.error,
      summary: run?.summary ?? null,
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
