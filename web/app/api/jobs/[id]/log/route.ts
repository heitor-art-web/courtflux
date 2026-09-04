import { canAccessJob } from "@/lib/supabase/access";
import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const access = await canAccessJob(id);
    if (!access.allowed) {
      return Response.json({ error: access.error }, { status: access.status });
    }
    const url = new URL(request.url);
    const tail = url.searchParams.get("tail") ?? "120";
    return proxyWorkerResponse(
      await workerFetch(`/jobs/${encodeURIComponent(id)}/log?tail=${encodeURIComponent(tail)}`),
    );
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
