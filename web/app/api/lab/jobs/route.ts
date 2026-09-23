import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { proxyWorkerResponse, workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

function directLabEnabled() {
  return process.env.NODE_ENV !== "production" || process.env.COURTFLUX_ENABLE_LAB_DIRECT === "1";
}

function emptyObject(value: unknown) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0);
}

async function canonicalMatchConfig() {
  // Repository-local config is a development fallback only. Vercel should never
  // trace or read repo-local data files into a serverless function bundle.
  if (process.env.VERCEL) {
    throw new Error("Canonical local match_config.json is unavailable on Vercel");
  }

  const candidates = [
    resolve(process.cwd(), "..", "data", "match_config.json"),
    resolve(process.cwd(), "data", "match_config.json"),
  ];

  let lastError: unknown;
  for (const path of candidates) {
    try {
      return JSON.parse(await readFile(/* turbopackIgnore: true */ path, "utf8")) as Record<string, unknown>;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Canonical match_config.json is unavailable");
}

export async function POST(request: Request) {
  if (!directLabEnabled()) {
    return Response.json({ error: "Direct Beta Lab worker access is disabled" }, { status: 404 });
  }

  try {
    const requestBody = (await request.json()) as Record<string, unknown>;
    if (requestBody.match_config == null || emptyObject(requestBody.match_config)) {
      requestBody.match_config = await canonicalMatchConfig();
    }

    // Worker-backed CourtFlux persistence is Supabase-only. Never allow the Lab UI
    // or an older cached client to reactivate the legacy local tracking.db path.
    requestBody.persist_results = false;
    delete requestBody.db_batch_size;

    // An explicit backend configured for the web Lab wins over stale worker health
    // defaults or an older browser state. This makes
    // `export BASKETBALL_JERSEY_BACKEND=... && npm run dev` deterministic.
    const webBackend = process.env.BASKETBALL_JERSEY_BACKEND?.trim();
    if (webBackend) {
      requestBody.jersey_backend = webBackend;
    }

    return proxyWorkerResponse(
      await workerFetch("/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      }),
    );
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
