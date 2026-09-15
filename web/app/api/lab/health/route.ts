import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

function directLabEnabled() {
  return process.env.NODE_ENV !== "production" || process.env.COURTFLUX_ENABLE_LAB_DIRECT === "1";
}

async function canonicalMatchConfig() {
  const candidates = [
    resolve(process.cwd(), "..", "data", "match_config.json"),
    resolve(process.cwd(), "data", "match_config.json"),
  ];

  let lastError: unknown;
  for (const path of candidates) {
    try {
      return JSON.parse(await readFile(path, "utf8")) as Record<string, unknown>;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Canonical match_config.json is unavailable");
}

export async function GET() {
  if (!directLabEnabled()) {
    return Response.json({ error: "Direct Beta Lab worker access is disabled" }, { status: 404 });
  }
  try {
    const workerResponse = await workerFetch("/health");
    if (!workerResponse.ok) return proxyWorkerResponse(workerResponse);

    const data = (await workerResponse.json()) as Record<string, unknown>;
    const labDefaults = {
      ...((data.lab_defaults as Record<string, unknown> | undefined) ?? {}),
    };

    // When the web Lab is deliberately started with a jersey backend override,
    // expose that same value to the UI. The jobs proxy enforces it as well, so
    // health display and submitted job can no longer disagree.
    const webBackend = process.env.BASKETBALL_JERSEY_BACKEND?.trim();
    if (webBackend) {
      labDefaults.jersey_backend = webBackend;
    }

    try {
      labDefaults.match_config = await canonicalMatchConfig();
      labDefaults.match_config_source = "data/match_config.json";
    } catch {
      // Keep the worker-provided config as a fallback when the web process cannot
      // access the repository checkout (for example, a packaged deployment).
    }

    data.lab_defaults = labDefaults;
    return Response.json(data, {
      status: workerResponse.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    return Response.json({ error: error instanceof Error ? error.message : String(error) }, { status: 503 });
  }
}
