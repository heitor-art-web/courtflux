const LOCAL_WORKER_URL = "http://127.0.0.1:8787";

export async function workerFetch(path: string, init: RequestInit = {}) {
  const baseUrl = (process.env.WORKER_API_URL ?? LOCAL_WORKER_URL).replace(/\/$/, "");
  const token = process.env.WORKER_API_TOKEN?.trim();

  const headers = new Headers(init.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  return fetch(`${baseUrl}${path}`, { ...init, headers, cache: "no-store" });
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
