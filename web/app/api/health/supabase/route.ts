import { NextResponse } from "next/server";
import { createAdminClient } from "@/lib/supabase/admin";

export const dynamic = "force-dynamic";

export async function GET() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const publishableKey = process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY;
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  const env = {
    url: Boolean(url),
    publishableKey: Boolean(publishableKey),
    serviceRoleKey: Boolean(serviceRoleKey),
  };

  if (!url || !publishableKey || !serviceRoleKey) {
    return NextResponse.json(
      { ok: false, env, publicApi: false, adminApi: false },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }

  try {
    const publicResponse = await fetch(`${url.replace(/\/$/, "")}/auth/v1/settings`, {
      headers: { apikey: publishableKey },
      cache: "no-store",
    });

    const admin = createAdminClient();
    const { error: adminError } = await admin
      .from("plans")
      .select("code", { head: true, count: "exact" });

    const publicApi = publicResponse.ok;
    const adminApi = !adminError;
    const ok = publicApi && adminApi;

    return NextResponse.json(
      { ok, env, publicApi, adminApi },
      {
        status: ok ? 200 : 503,
        headers: { "Cache-Control": "no-store" },
      },
    );
  } catch {
    return NextResponse.json(
      { ok: false, env, publicApi: false, adminApi: false },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
