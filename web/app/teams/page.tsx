import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { TeamManager } from "./team-manager";

export const dynamic = "force-dynamic";

export default async function TeamsPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const { data: membership, error: membershipError } = await supabase
    .from("workspace_members")
    .select("workspace_id, role, workspaces(name, primary_team_id)")
    .eq("user_id", user.id)
    .eq("status", "active")
    .order("created_at", { ascending: true })
    .limit(1)
    .maybeSingle();

  if (membershipError || !membership?.workspace_id) redirect("/account");

  const workspace = Array.isArray(membership.workspaces)
    ? membership.workspaces[0]
    : membership.workspaces;
  const workspaceId = membership.workspace_id as string;

  const [teamsResult, playersResult] = await Promise.all([
    supabase
      .from("teams")
      .select("id, name, short_name, season, age_group, is_archived, archived_at, created_at")
      .eq("workspace_id", workspaceId)
      .order("is_archived", { ascending: true })
      .order("name", { ascending: true })
      .limit(100),
    supabase
      .from("team_players")
      .select("id, team_id, name, jersey_number, position, is_active")
      .eq("workspace_id", workspaceId)
      .eq("is_active", true)
      .order("jersey_number", { ascending: true, nullsFirst: false })
      .order("name", { ascending: true })
      .limit(1000),
  ]);

  const playersByTeam = new Map<string, Array<{
    id: string;
    name: string;
    jerseyNumber: number | null;
    position: string | null;
  }>>();

  for (const player of playersResult.data ?? []) {
    const list = playersByTeam.get(player.team_id) ?? [];
    list.push({
      id: player.id,
      name: player.name,
      jerseyNumber: player.jersey_number,
      position: player.position,
    });
    playersByTeam.set(player.team_id, list);
  }

  return (
    <TeamManager
      workspaceName={workspace?.name ?? "CourtFlux workspace"}
      role={membership.role ?? "member"}
      primaryTeamId={workspace?.primary_team_id ?? null}
      initialTeams={(teamsResult.data ?? []).map((team) => ({
        id: team.id,
        name: team.name,
        shortName: team.short_name,
        season: team.season,
        ageGroup: team.age_group,
        isArchived: team.is_archived,
        roster: playersByTeam.get(team.id) ?? [],
      }))}
    />
  );
}
