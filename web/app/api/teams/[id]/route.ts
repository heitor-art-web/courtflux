import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

function cleanText(value: unknown, maxLength: number) {
  if (typeof value !== "string") return null;
  const clean = value.trim().replace(/\s+/g, " ");
  return clean ? clean.slice(0, maxLength) : null;
}

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return Response.json({ error: "Authentication required" }, { status: 401 });

  const { data: membership, error: membershipError } = await supabase
    .from("workspace_members")
    .select("workspace_id, role")
    .eq("user_id", user.id)
    .eq("status", "active")
    .order("created_at", { ascending: true })
    .limit(1)
    .maybeSingle();

  if (membershipError) return Response.json({ error: membershipError.message }, { status: 500 });
  if (!membership?.workspace_id) return Response.json({ error: "No active CourtFlux workspace" }, { status: 403 });
  if (!["owner", "admin", "analyst"].includes(membership.role)) {
    return Response.json({ error: "Your workspace role cannot manage teams" }, { status: 403 });
  }

  const { data: team } = await supabase
    .from("teams")
    .select("id")
    .eq("id", id)
    .eq("workspace_id", membership.workspace_id)
    .maybeSingle();
  if (!team) return Response.json({ error: "Team not found" }, { status: 404 });

  const payload = (await request.json()) as Record<string, unknown>;
  const { data: workspace } = await supabase
    .from("workspaces")
    .select("primary_team_id")
    .eq("id", membership.workspace_id)
    .maybeSingle();

  if (payload.is_archived === true && workspace?.primary_team_id === id && !["owner", "admin"].includes(membership.role)) {
    return Response.json({ error: "Only workspace owners and admins can archive the primary team" }, { status: 403 });
  }

  const updates: Record<string, unknown> = {};
  if ("name" in payload) {
    const name = cleanText(payload.name, 120);
    if (!name) return Response.json({ error: "Team name is required" }, { status: 400 });
    updates.name = name;
  }
  if ("short_name" in payload) updates.short_name = cleanText(payload.short_name, 40);
  if ("season" in payload) updates.season = cleanText(payload.season, 80);
  if ("age_group" in payload) updates.age_group = cleanText(payload.age_group, 80);
  if (typeof payload.is_archived === "boolean") {
    updates.is_archived = payload.is_archived;
    updates.archived_at = payload.is_archived ? new Date().toISOString() : null;
  }

  if (Object.keys(updates).length) {
    const { error } = await supabase
      .from("teams")
      .update(updates)
      .eq("id", id)
      .eq("workspace_id", membership.workspace_id);
    if (error) {
      const duplicate = error.code === "23505";
      return Response.json(
        { error: duplicate ? "A team with this name already exists in the workspace" : error.message },
        { status: duplicate ? 409 : 500 },
      );
    }
  }

  if (payload.make_primary === true) {
    if (!["owner", "admin"].includes(membership.role)) {
      return Response.json({ error: "Only workspace owners and admins can change the primary team" }, { status: 403 });
    }
    const { error } = await supabase
      .from("workspaces")
      .update({ primary_team_id: id })
      .eq("id", membership.workspace_id);
    if (error) return Response.json({ error: error.message }, { status: 500 });
  }

  if (payload.is_archived === true && workspace?.primary_team_id === id) {
    const { error } = await supabase
      .from("workspaces")
      .update({ primary_team_id: null })
      .eq("id", membership.workspace_id);
    if (error) return Response.json({ error: error.message }, { status: 500 });
  }

  return Response.json({ ok: true });
}
