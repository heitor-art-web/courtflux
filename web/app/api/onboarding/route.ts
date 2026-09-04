import { createClient } from "@/lib/supabase/server";
import { isOnboardingRole } from "@/lib/onboarding";

export const dynamic = "force-dynamic";

function cleanText(value: unknown, maxLength: number) {
  if (typeof value !== "string") return "";
  return value.trim().replace(/\s+/g, " ").slice(0, maxLength);
}

export async function POST(request: Request) {
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
      .select("workspace_id, role, workspaces(name)")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();

    if (membershipError) throw membershipError;
    if (!membership?.workspace_id) {
      return Response.json({ error: "No active CourtFlux workspace" }, { status: 403 });
    }
    if (!new Set(["owner", "admin"]).has(membership.role)) {
      return Response.json({ error: "Only a workspace owner or admin can complete onboarding" }, { status: 403 });
    }

    const body = (await request.json()) as {
      basketball_role?: unknown;
      team_name?: unknown;
      workspace_name?: unknown;
    };

    if (!isOnboardingRole(body.basketball_role)) {
      return Response.json({ error: "Choose your basketball role" }, { status: 400 });
    }

    const teamName = cleanText(body.team_name, 80);
    const requestedWorkspaceName = cleanText(body.workspace_name, 80);
    const workspaceName = requestedWorkspaceName || `${teamName} Workspace`;

    if (teamName.length < 2) {
      return Response.json({ error: "Add the team you work with most" }, { status: 400 });
    }
    if (workspaceName.length < 2) {
      return Response.json({ error: "Add a workspace name" }, { status: 400 });
    }

    const workspaceId = membership.workspace_id as string;
    const { data: team, error: teamError } = await supabase
      .from("teams")
      .upsert(
        {
          workspace_id: workspaceId,
          name: teamName,
        },
        { onConflict: "workspace_id,name" },
      )
      .select("id, name")
      .single();

    if (teamError || !team?.id) throw teamError ?? new Error("Could not save the first team");

    const { error: profileError } = await supabase
      .from("profiles")
      .update({ basketball_role: body.basketball_role })
      .eq("id", user.id);
    if (profileError) throw profileError;

    const { error: workspaceError } = await supabase
      .from("workspaces")
      .update({
        name: workspaceName,
        primary_team_id: team.id,
        onboarding_completed_at: new Date().toISOString(),
      })
      .eq("id", workspaceId);
    if (workspaceError) throw workspaceError;

    return Response.json({
      ok: true,
      workspace: { id: workspaceId, name: workspaceName },
      team,
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
