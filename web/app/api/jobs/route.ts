import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const body = await request.text();
    const upstream = await workerFetch("/jobs", {
      method: "POST",
      body,
    });
    return proxyWorkerResponse(upstream);
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
