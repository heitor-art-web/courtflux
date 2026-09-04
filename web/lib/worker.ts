export async function workerFetch(path: string, init: RequestInit = {}) {
  const baseUrl = process.env.WORKER_API_URL?.replace(/\/$/, "");
  const token = process.env.WORKER_API_TOKEN;

  if (!baseUrl || !token) {
    throw new Error("WORKER_API_URL and WORKER_API_TOKEN must be configured in Vercel");
  }

  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  return fetch(`${baseUrl}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });
}

export async function proxyWorkerResponse(response: Response) {
  const body = await response.text();
  return new Response(body, {
    status: response.status,
    headers: {
      "Content-Type": response.headers.get("Content-Type") ?? "application/json",
      "Cache-Control": "no-store",
    },
  });
}
