import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import {
  canStartAnalysis,
  normalizePlanCode,
  resolveMonthlyAllowance,
} from "@/lib/usage";
import { workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

type GameContext = {
  home_team?: unknown;
  away_team?: unknown;
  game_date?: unknown;
  competition?: unknown;
  source_type?: unknown;
  source_label?: unknown;
  home_kit_colors?: unknown;
  away_kit_colors?: unknown;
};

type UsageReservationResult = {
  granted?: boolean;
  used_minutes?: number;
  reserved_minutes?: number;
  remaining_minutes?: number | null;
  status?: string;
  reason?: string;
};

function numberFromUnknown(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function textFromUnknown(value: unknown, maxLength = 160) {
  if (typeof value !== "string") return null;
  const clean = value.trim();
  return clean ? clean.slice(0, maxLength) : null;
}

function colorList(value: unknown) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 8);
}

function reservationFromUnknown(value: unknown): UsageReservationResult {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  return value as UsageReservationResult;
}

async function releaseUsageReservation(
  admin: ReturnType<typeof createAdminClient>,
  runId: string | null,
  reason: string,
) {
  if (!runId) return;
  const { error } = await admin.rpc("release_analysis_minutes_reservation", {
    p_run_id: runId,
    p_reason: reason.slice(0, 500),
  });
  if (error) {
    console.error("Could not release analysis usage reservation", error);
  }
}

async function ensureTeam(
  admin: ReturnType<typeof createAdminClient>,
  workspaceId: string,
  name: string,
) {
  const { data, error } = await admin
    .from("teams")
    .upsert(
      { workspace_id: workspaceId, name },
      { onConflict: "workspace_id,name" },
    )
    .select("id")
    .single();
  if (error) throw error;
  return data.id as string;
}

export async function POST(request: Request) {
  let runId: string | null = null;
  let jobId: string | null = null;
  let usageReservationActive = false;

  try {
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return Response.json({ error: "Authentication required" }, { status: 401 });
    }

    const { data: membership, error: membershipError } = await supabase
      .from("workspace_members")
      .select("workspace_id, role")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();

    if (membershipError) throw membershipError;
    if (!membership?.workspace_id) {
      return Response.json({ error: "No active CourtFlux workspace" }, { status: 403 });
    }
    if (!["owner", "admin", "analyst"].includes(membership.role)) {
      return Response.json({ error: "Your workspace role cannot start analyses" }, { status: 403 });
    }

    const payload = (await request.json()) as Record<string, unknown>;
    const workspaceId = membership.workspace_id as string;
    const admin = createAdminClient();
    const startTimeSec = numberFromUnknown(payload.start_time_sec) ?? 0;
    const endTimeSec = numberFromUnknown(payload.end_time_sec);

    if (startTimeSec < 0 || endTimeSec === null || endTimeSec <= startTimeSec) {
      return Response.json({ error: "Analysis end time must be greater than start time" }, { status: 400 });
    }
    const requestedMinutes = (endTimeSec - startTimeSec) / 60;

    const { data: entitlement } = await admin
      .from("workspace_entitlements")
      .select("plan_code, status, features, limits")
      .eq("workspace_id", workspaceId)
      .maybeSingle();

    const now = new Date();
    const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
    const nextMonth = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));

    const [usageResult, grantsResult] = await Promise.all([
      admin
        .from("workspace_usage_monthly")
        .select("quantity")
        .eq("workspace_id", workspaceId)
        .eq("metric_code", "analysis_minutes")
        .gte("usage_month", monthStart.toISOString())
        .lt("usage_month", nextMonth.toISOString())
        .maybeSingle(),
      admin
        .from("usage_allowance_grants")
        .select("quantity, valid_from, valid_until")
        .eq("workspace_id", workspaceId)
        .eq("metric_code", "analysis_minutes")
        .limit(100),
    ]);

    const usedMinutes = numberFromUnknown(usageResult.data?.quantity) ?? 0;
    const nowMs = now.getTime();
    const grantedMinutes = grantsResult.error
      ? 0
      : (grantsResult.data ?? []).reduce((total, grant) => {
          const startsAt = new Date(grant.valid_from).getTime();
          const endsAt = grant.valid_until ? new Date(grant.valid_until).getTime() : null;
          const active = Number.isFinite(startsAt) && startsAt <= nowMs && (endsAt === null || endsAt > nowMs);
          return active ? total + (numberFromUnknown(grant.quantity) ?? 0) : total;
        }, 0);

    const planCode = normalizePlanCode(entitlement?.plan_code);
    const limits = (entitlement?.limits ?? {}) as Record<string, unknown>;
    const monthlyAllowance = resolveMonthlyAllowance(planCode, limits, grantedMinutes);
    const remainingMinutes = monthlyAllowance === null ? null : Math.max(0, monthlyAllowance - usedMinutes);

    if (!canStartAnalysis(usedMinutes, monthlyAllowance)) {
      return Response.json(
        {
          error: "Monthly analysis allowance reached",
          code: "analysis_allowance_exhausted",
          plan_code: planCode,
          analysis_minutes_used: usedMinutes,
          analysis_minutes_allowance: monthlyAllowance,
          analysis_minutes_remaining: 0,
          billing_active: false,
          detail: "New analyses are paused until the monthly allowance resets or a CourtFlux admin grants additional beta minutes.",
        },
        { status: 429 },
      );
    }

    if (remainingMinutes !== null && requestedMinutes > remainingMinutes + 0.001) {
      return Response.json(
        {
          error: "Not enough analysis minutes remaining for this video window",
          code: "analysis_allowance_insufficient",
          plan_code: planCode,
          analysis_minutes_requested: requestedMinutes,
          analysis_minutes_used: usedMinutes,
          analysis_minutes_allowance: monthlyAllowance,
          analysis_minutes_remaining: remainingMinutes,
          billing_active: false,
          detail: "Shorten the analysis window or ask a CourtFlux admin for additional beta minutes. No overage will be charged automatically.",
        },
        { status: 429 },
      );
    }

    let gameId: string | null = null;
    const context = (payload.game_context && typeof payload.game_context === "object")
      ? payload.game_context as GameContext
      : null;

    if (context) {
      const homeTeam = textFromUnknown(context.home_team);
      const awayTeam = textFromUnknown(context.away_team);
      if (!homeTeam || !awayTeam || homeTeam === awayTeam) {
        return Response.json({ error: "New Analysis requires two different team names" }, { status: 400 });
      }

      const [homeTeamId, awayTeamId] = await Promise.all([
        ensureTeam(admin, workspaceId, homeTeam),
        ensureTeam(admin, workspaceId, awayTeam),
      ]);
      const rawDate = textFromUnknown(context.game_date, 32);
      const gameDate = rawDate && /^\d{4}-\d{2}-\d{2}$/.test(rawDate)
        ? `${rawDate}T12:00:00.000Z`
        : null;
      const requestedSourceType = textFromUnknown(context.source_type, 24);
      const sourceType = requestedSourceType === "youtube"
        ? "youtube"
        : requestedSourceType === "upload"
          ? "upload"
          : "video";
      const source = textFromUnknown(payload.video_source, 2048);

      const { data: game, error: gameError } = await admin
        .from("games")
        .insert({
          workspace_id: workspaceId,
          created_by: user.id,
          title: `${homeTeam} vs ${awayTeam}`,
          competition: textFromUnknown(context.competition),
          game_date: gameDate,
          home_team_id: homeTeamId,
          away_team_id: awayTeamId,
          video_source: source,
          source_type: sourceType,
          metadata: {
            source_label: textFromUnknown(context.source_label, 240),
            home_kit_colors: colorList(context.home_kit_colors),
            away_kit_colors: colorList(context.away_kit_colors),
          },
        })
        .select("id")
        .single();
      if (gameError) throw gameError;
      gameId = game.id as string;
    }

    runId = crypto.randomUUID();
    jobId = crypto.randomUUID();

    const { error: runError } = await admin.from("analysis_runs").insert({
      id: runId,
      workspace_id: workspaceId,
      game_id: gameId,
      created_by: user.id,
      status: "queued",
      video_source: typeof payload.video_source === "string" ? payload.video_source : null,
      start_time_sec: startTimeSec,
      end_time_sec: endTimeSec,
      parameters: payload,
    });
    if (runError) throw runError;

    const { error: jobError } = await admin.from("analysis_jobs").insert({
      id: jobId,
      run_id: runId,
      workspace_id: workspaceId,
      status: "queued",
      request: payload,
    });
    if (jobError) throw jobError;

    const { data: reservationData, error: reservationError } = await admin.rpc(
      "reserve_analysis_minutes",
      {
        p_workspace_id: workspaceId,
        p_run_id: runId,
        p_user_id: user.id,
        p_quantity: requestedMinutes,
        p_allowance_minutes: monthlyAllowance,
      },
    );
    if (reservationError) throw reservationError;

    const reservation = reservationFromUnknown(reservationData);
    if (!reservation.granted) {
      const errorText = "Analysis allowance changed before the requested minutes could be reserved";
      const completedAt = new Date().toISOString();
      await Promise.all([
        admin.from("analysis_jobs").update({ status: "failed", error: errorText, completed_at: completedAt }).eq("id", jobId),
        admin.from("analysis_runs").update({ status: "failed", completed_at: completedAt }).eq("id", runId),
      ]);
      return Response.json(
        {
          error: "Not enough unreserved analysis minutes remaining for this video window",
          code: "analysis_allowance_reserved_elsewhere",
          plan_code: planCode,
          analysis_minutes_requested: requestedMinutes,
          analysis_minutes_used: reservation.used_minutes ?? usedMinutes,
          analysis_minutes_reserved: reservation.reserved_minutes ?? null,
          analysis_minutes_allowance: monthlyAllowance,
          analysis_minutes_remaining: reservation.remaining_minutes ?? 0,
          billing_active: false,
          detail: "Another analysis may already be using the remaining allowance. Shorten the window or wait for an active analysis to finish or fail.",
        },
        { status: 429 },
      );
    }
    usageReservationActive = true;

    const upstream = await workerFetch("/jobs", {
      method: "POST",
      body: JSON.stringify({
        ...payload,
        start_time_sec: startTimeSec,
        end_time_sec: endTimeSec,
        platform_run_id: runId,
        platform_job_id: jobId,
        workspace_id: workspaceId,
        user_id: user.id,
        subscription_plan: planCode,
      }),
    });

    const text = await upstream.text();
    let data: Record<string, unknown> = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }

    if (!upstream.ok) {
      const errorText = String(data.detail ?? data.error ?? `Worker HTTP ${upstream.status}`);
      const completedAt = new Date().toISOString();
      await releaseUsageReservation(admin, runId, "worker_dispatch_failed");
      usageReservationActive = false;
      await Promise.all([
        admin.from("analysis_jobs").update({ status: "failed", error: errorText, completed_at: completedAt }).eq("id", jobId),
        admin.from("analysis_runs").update({ status: "failed", completed_at: completedAt }).eq("id", runId),
      ]);
      return Response.json({ ...data, run_id: runId, job_id: jobId, game_id: gameId }, { status: upstream.status });
    }

    return Response.json({
      ...data,
      run_id: runId,
      job_id: jobId,
      game_id: gameId,
      analysis_minutes_estimated: requestedMinutes,
      analysis_minutes_reserved: requestedMinutes,
      analysis_minutes_remaining_before: remainingMinutes,
      analysis_minutes_remaining_after_reservation: reservation.remaining_minutes ?? null,
    });
  } catch (error) {
    const errorText = error instanceof Error ? error.message : String(error);
    try {
      if (runId || jobId) {
        const admin = createAdminClient();
        if (usageReservationActive) {
          await releaseUsageReservation(admin, runId, "job_creation_or_dispatch_error");
          usageReservationActive = false;
        }
        const completedAt = new Date().toISOString();
        const writes = [];
        if (jobId) writes.push(admin.from("analysis_jobs").update({ status: "failed", error: errorText, completed_at: completedAt }).eq("id", jobId));
        if (runId) writes.push(admin.from("analysis_runs").update({ status: "failed", completed_at: completedAt }).eq("id", runId));
        await Promise.all(writes);
      }
    } catch {
      // Preserve the original request error.
    }
    return Response.json({ error: errorText }, { status: 503 });
  }
}
