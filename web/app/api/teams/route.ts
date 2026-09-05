import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

function cleanText(value: unknown, maxLength: number) {
  if (typeof value !== "string") return null;
  const clean = value.trim().replace(/\s+/g, " ");
  return clean ? clean.slice(0, maxLength) : null;
}

export async function POST(request: Request) {
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

  const payload = (await request.json()) as Record<string, unknown>;
  const name = cleanText(payload.name, 120);
  if (!name) return Response.json({ error: "Team name is required" }, { status: 400 });

  const { data, error } = await supabase
    .from("teams")
    .insert({
      workspace_id: membership.workspace_id,
      name,
      short_name: cleanText(payload.short_name, 40),
      season: cleanText(payload.season, 80),
      age_group: cleanText(payload.age_group, 80),
      is_archived: false,
      archived_at: null,
    })
    .select("id")
    .single();

  if (error) {
    const duplicate = error.code === "23505";
    return Response.json(
      { error: duplicate ? "A team with this name already exists in the workspace" : error.message },
      { status: duplicate ? 409 : 500 },
    );
  }

  return Response.json({ id: data.id }, { status: 201 });
}
