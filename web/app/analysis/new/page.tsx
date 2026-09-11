import { redirect } from "next/navigation";
import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import {
  getUsageLevel,
  normalizePlanCode,
  resolveMonthlyAllowance,
} from "@/lib/usage";
import { NewAnalysisForm, type RetryDraft } from "./new-analysis-form";

export const dynamic = "force-dynamic";

function numberFromUnknown(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function recordFromUnknown(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function stringFromUnknown(value: unknown) {
  return typeof value === "string" ? value : "";
}

function stringArrayFromUnknown(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function rosterText(value: unknown) {
  const roster = recordFromUnknown(value);
  return Object.entries(roster)
    .map(([jersey, player]) => {
      const name = stringFromUnknown(recordFromUnknown(player).name);
      return `${jersey} ${name || `Player #${jersey}`}`;
    })
    .join("\n");
}

function uploadObjectPath(value: string) {
  const prefix = "supabase://analysis-videos/";
  return value.startsWith(prefix) ? value.slice(prefix.length) : null;
}

export default async function NewAnalysisPage({
  searchParams,
}: {
  searchParams: Promise<{ retry?: string | string[] }>;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) redirect("/login");

  const { data: membership, error: membershipError } = await supabase
    .from("workspace_members")
    .select("workspace_id, role, workspaces(name)")
    .eq("user_id", user.id)
    .eq("status", "active")
    .order("created_at", { ascending: true })
    .limit(1)
    .maybeSingle();

  if (membershipError || !membership?.workspace_id) redirect("/account");

  const workspaceId = membership.workspace_id as string;
  const admin = createAdminClient();
  const now = new Date();
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  const nextMonth = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));

  const [entitlementResult, usageResult, grantsResult, teamsResult, playersResult] = await Promise.all([
    admin
      .from("workspace_entitlements")
      .select("plan_code, plan_name, limits")
      .eq("workspace_id", workspaceId)
      .maybeSingle(),
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
    admin
      .from("teams")
      .select("id, name, short_name")
      .eq("workspace_id", workspaceId)
      .eq("is_archived", false)
      .order("name")
      .limit(50),
    admin
      .from("team_players")
      .select("team_id, name, jersey_number, position")
      .eq("workspace_id", workspaceId)
      .eq("is_active", true)
      .order("jersey_number", { ascending: true, nullsFirst: false })
      .order("name", { ascending: true })
      .limit(1000),
  ]);

  const nowMs = now.getTime();
  const grantedMinutes = grantsResult.error
    ? 0
    : (grantsResult.data ?? []).reduce((total, grant) => {
        const startsAt = new Date(grant.valid_from).getTime();
        const endsAt = grant.valid_until ? new Date(grant.valid_until).getTime() : null;
        const active = Number.isFinite(startsAt) && startsAt <= nowMs && (endsAt === null || endsAt > nowMs);
        return active ? total + (numberFromUnknown(grant.quantity) ?? 0) : total;
      }, 0);

  const entitlement = entitlementResult.data;
  const planCode = normalizePlanCode(entitlement?.plan_code);
  const planName = entitlement?.plan_name || (planCode === "beta" ? "Beta" : planCode);
  const usedMinutes = numberFromUnknown(usageResult.data?.quantity) ?? 0;
  const allowance = resolveMonthlyAllowance(
    planCode,
    (entitlement?.limits ?? {}) as Record<string, unknown>,
    grantedMinutes,
  );
  const workspace = Array.isArray(membership.workspaces)
    ? membership.workspaces[0]
    : membership.workspaces;

  const rosterByTeam = new Map<string, Array<{
    name: string;
    jerseyNumber: number | null;
    position: string | null;
  }>>();

  for (const player of playersResult.data ?? []) {
    const roster = rosterByTeam.get(player.team_id) ?? [];
    roster.push({
      name: player.name,
      jerseyNumber: player.jersey_number,
      position: player.position,
    });
    rosterByTeam.set(player.team_id, roster);
  }

  let retryDraft: RetryDraft | null = null;
  const params = await searchParams;
  const retryValue = Array.isArray(params.retry) ? params.retry[0] : params.retry;
  const retryId = retryValue && /^[0-9a-f]{8}-[0-9a-f-]{27,}$/i.test(retryValue) ? retryValue : null;

  if (retryId) {
    const { data: previousRun } = await admin
      .from("analysis_runs")
      .select("id, status, video_source, start_time_sec, end_time_sec, parameters")
      .eq("workspace_id", workspaceId)
      .eq("id", retryId)
      .in("status", ["failed", "canceled"])
      .maybeSingle();

    if (previousRun) {
      const parameters = recordFromUnknown(previousRun.parameters);
      const context = recordFromUnknown(parameters.game_context);
      const matchConfig = recordFromUnknown(parameters.match_config);
      const teamsConfig = recordFromUnknown(matchConfig.teams);
      const homeTeam = stringFromUnknown(context.home_team) || stringFromUnknown(recordFromUnknown(matchConfig.metadata).home_team);
      const awayTeam = stringFromUnknown(context.away_team) || stringFromUnknown(recordFromUnknown(matchConfig.metadata).away_team);
      const homeConfig = recordFromUnknown(teamsConfig[homeTeam]);
      const awayConfig = recordFromUnknown(teamsConfig[awayTeam]);
      const homeKits = recordFromUnknown(homeConfig.kits);
      const awayKits = recordFromUnknown(awayConfig.kits);
      const source = stringFromUnknown(previousRun.video_source);
      const sourceType = stringFromUnknown(context.source_type) === "youtube" || /^https?:\/\//i.test(source) ? "youtube" : "upload";
      let reusableSource = sourceType === "youtube";

      if (sourceType === "upload") {
        const objectPath = uploadObjectPath(source);
        if (objectPath) {
          const { data: upload } = await admin
            .from("analysis_video_uploads")
            .select("deleted_at, expires_at")
            .eq("workspace_id", workspaceId)
            .eq("object_path", objectPath)
            .maybeSingle();
          const expiresAt = upload?.expires_at ? new Date(upload.expires_at).getTime() : null;
          reusableSource = Boolean(upload && !upload.deleted_at && (expiresAt === null || expiresAt > nowMs));
        }
      }

      retryDraft = {
        runId: previousRun.id,
        sourceMode: sourceType,
        source: reusableSource ? source : "",
        sourceLabel: stringFromUnknown(context.source_label) || (sourceType === "youtube" ? "YouTube" : "Previously uploaded video"),
        reusableSource,
        startMinutes: String((numberFromUnknown(previousRun.start_time_sec) ?? 0) / 60),
        endMinutes: String((numberFromUnknown(previousRun.end_time_sec) ?? 60) / 60),
        homeTeam,
        awayTeam,
        gameDate: stringFromUnknown(context.game_date),
        competition: stringFromUnknown(context.competition),
        homeColors: stringArrayFromUnknown(context.home_kit_colors).join(", ") || stringArrayFromUnknown(homeKits.current).join(", "),
        awayColors: stringArrayFromUnknown(context.away_kit_colors).join(", ") || stringArrayFromUnknown(awayKits.current).join(", "),
        homeRoster: rosterText(homeConfig.roster),
        awayRoster: rosterText(awayConfig.roster),
        enableJersey: parameters.enable_jersey === true,
      };
    }
  }

  return (
    <NewAnalysisForm
      workspaceName={workspace?.name ?? "CourtFlux workspace"}
      role={membership.role ?? "member"}
      planName={planName}
      usedMinutes={usedMinutes}
      allowance={allowance}
      usageLevel={getUsageLevel(usedMinutes, allowance)}
      initialDraft={retryDraft}
      teams={(teamsResult.data ?? []).map((team) => ({
        id: team.id,
        name: team.name,
        shortName: team.short_name,
        roster: rosterByTeam.get(team.id) ?? [],
      }))}
    />
  );
}
