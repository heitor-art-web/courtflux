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

export async function POST(request: Request) {
  if (!directLabEnabled()) {
    return Response.json({ error: "Direct Beta Lab worker access is disabled" }, { status: 404 });
  }

  try {
    const requestBody = (await request.json()) as Record<string, unknown>;
    if (requestBody.match_config == null || emptyObject(requestBody.match_config)) {
      requestBody.match_config = await canonicalMatchConfig();
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
