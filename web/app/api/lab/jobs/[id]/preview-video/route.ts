import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

function directLabEnabled() {
  return process.env.NODE_ENV !== "production" || process.env.COURTFLUX_ENABLE_LAB_DIRECT === "1";
}

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  if (!directLabEnabled()) return Response.json({ error: "Direct Beta Lab worker access is disabled" }, { status: 404 });
  try {
    const { id } = await context.params;
    const range = request.headers.get("range");
    return proxyWorkerResponse(
      await workerFetch(`/jobs/${encodeURIComponent(id)}/preview-video`, {
        headers: range ? { Range: range } : undefined,
      }),
    );
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : String(error) }, { status: 503 });
  }
}
