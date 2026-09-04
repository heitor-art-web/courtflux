import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return proxyWorkerResponse(await workerFetch("/health"));
  } catch (error) {
    return Response.json(
      { status: "error", error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
