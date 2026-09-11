const LOCAL_WORKER_URL = "http://127.0.0.1:8787";

function workerBaseUrl() {
  const configured = process.env.WORKER_API_URL?.trim();
  if (configured) return configured.replace(/\/$/, "");
  if (process.env.NODE_ENV !== "production") return LOCAL_WORKER_URL;
  throw new Error("CourtFlux analysis worker is not configured for this environment");
}

export async function workerFetch(path: string, init: RequestInit = {}) {
  const baseUrl = workerBaseUrl();
  const token = process.env.WORKER_API_TOKEN?.trim();

  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  try {
    return await fetch(`${baseUrl}${path}`, { ...init, headers, cache: "no-store" });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`CourtFlux analysis worker is unreachable: ${detail}`);
  }
}

export async function proxyWorkerResponse(response: Response) {
  const body = await response.arrayBuffer();
  const headers = new Headers();
  headers.set("Content-Type", response.headers.get("Content-Type") ?? "application/octet-stream");
  headers.set("Cache-Control", "no-store");
  const disposition = response.headers.get("Content-Disposition");
  if (disposition) headers.set("Content-Disposition", disposition);
  return new Response(body, { status: response.status, headers });
}
