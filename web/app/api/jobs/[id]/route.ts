import { canAccessJob } from "@/lib/supabase/access";
import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const access = await canAccessJob(id);
    if (!access.allowed) {
      return Response.json({ error: access.error }, { status: access.status });
    }
    return proxyWorkerResponse(await workerFetch(`/jobs/${encodeURIComponent(id)}`));
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
