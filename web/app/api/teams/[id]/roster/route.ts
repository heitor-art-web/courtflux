import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

type RosterInput = {
  id?: unknown;
  name?: unknown;
  jersey_number?: unknown;
  position?: unknown;
};

function cleanText(value: unknown, maxLength: number) {
  if (typeof value !== "string") return null;
  const clean = value.trim().replace(/\s+/g, " ");
  return clean ? clean.slice(0, maxLength) : null;
}

function jerseyNumber(value: unknown) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = typeof value === "number" ? value : Number(value);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > 999) return undefined;
  return parsed;
}

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id: teamId } = await params;
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
    return Response.json({ error: "Your workspace role cannot manage rosters" }, { status: 403 });
  }

  const { data: team } = await supabase
    .from("teams")
    .select("id")
    .eq("id", teamId)
    .eq("workspace_id", membership.workspace_id)
    .maybeSingle();
  if (!team) return Response.json({ error: "Team not found" }, { status: 404 });

  const payload = (await request.json()) as { players?: unknown };
  if (!Array.isArray(payload.players) || payload.players.length > 60) {
    return Response.json({ error: "Roster must contain at most 60 players" }, { status: 400 });
  }

  const players = payload.players as RosterInput[];
  const cleaned = [] as Array<{
    id: string | null;
    name: string;
    jersey_number: number | null;
    position: string | null;
  }>;
  const seenJerseys = new Set<number>();

  for (const player of players) {
    const name = cleanText(player.name, 120);
    if (!name) return Response.json({ error: "Each roster entry needs a player name" }, { status: 400 });
    const jersey = jerseyNumber(player.jersey_number);
    if (jersey === undefined) return Response.json({ error: "Jersey numbers must be between 0 and 999" }, { status: 400 });
    if (jersey !== null) {
      if (seenJerseys.has(jersey)) return Response.json({ error: `Jersey #${jersey} appears more than once` }, { status: 400 });
      seenJerseys.add(jersey);
    }
    cleaned.push({
      id: typeof player.id === "string" && player.id ? player.id : null,
      name,
      jersey_number: jersey,
      position: cleanText(player.position, 80),
    });
  }

  const { data: existing, error: existingError } = await supabase
    .from("team_players")
    .select("id")
    .eq("workspace_id", membership.workspace_id)
    .eq("team_id", teamId);
  if (existingError) return Response.json({ error: existingError.message }, { status: 500 });

  const existingIds = new Set((existing ?? []).map((item) => item.id));
  const retainedIds = new Set(cleaned.map((item) => item.id).filter((value): value is string => Boolean(value)));
  const invalidId = [...retainedIds].find((playerId) => !existingIds.has(playerId));
  if (invalidId) return Response.json({ error: "Roster contains a player that does not belong to this team" }, { status: 400 });

  const removedIds = [...existingIds].filter((playerId) => !retainedIds.has(playerId));
  if (removedIds.length) {
    const { error } = await supabase
      .from("team_players")
      .update({ is_active: false })
      .eq("workspace_id", membership.workspace_id)
      .eq("team_id", teamId)
      .in("id", removedIds);
    if (error) return Response.json({ error: error.message }, { status: 500 });
  }

  // Temporarily clear active jersey numbers so swaps such as #7 ↔ #8 do not hit the partial unique index.
  if (retainedIds.size) {
    const { error } = await supabase
      .from("team_players")
      .update({ jersey_number: null })
      .eq("workspace_id", membership.workspace_id)
      .eq("team_id", teamId)
      .in("id", [...retainedIds]);
    if (error) return Response.json({ error: error.message }, { status: 500 });
  }

  for (const player of cleaned) {
    if (player.id) {
      const { error } = await supabase
        .from("team_players")
        .update({
          name: player.name,
          jersey_number: player.jersey_number,
          position: player.position,
          is_active: true,
        })
        .eq("id", player.id)
        .eq("workspace_id", membership.workspace_id)
        .eq("team_id", teamId);
      if (error) return Response.json({ error: error.message }, { status: 500 });
    } else {
      const { error } = await supabase.from("team_players").insert({
        workspace_id: membership.workspace_id,
        team_id: teamId,
        name: player.name,
        jersey_number: player.jersey_number,
        position: player.position,
        is_active: true,
      });
      if (error) return Response.json({ error: error.message }, { status: 500 });
    }
  }

  return Response.json({ ok: true });
}
