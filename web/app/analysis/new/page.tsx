import { redirect } from "next/navigation";
import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import {
  getUsageLevel,
  normalizePlanCode,
  resolveMonthlyAllowance,
} from "@/lib/usage";
import { NewAnalysisForm } from "./new-analysis-form";

export const dynamic = "force-dynamic";

function numberFromUnknown(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export default async function NewAnalysisPage() {
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

  return (
    <NewAnalysisForm
      workspaceName={workspace?.name ?? "CourtFlux workspace"}
      role={membership.role ?? "member"}
      planName={planName}
      usedMinutes={usedMinutes}
      allowance={allowance}
      usageLevel={getUsageLevel(usedMinutes, allowance)}
      teams={(teamsResult.data ?? []).map((team) => ({
        id: team.id,
        name: team.name,
        shortName: team.short_name,
        roster: rosterByTeam.get(team.id) ?? [],
      }))}
    />
  );
}
