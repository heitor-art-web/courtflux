import { parseHumanReviewCommand } from "@/lib/p3/human-review";
import type { HumanReviewRecord } from "@/lib/p3/human-review";
import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

function requestedDerivationId(request: Request) {
  const value = new URL(request.url).searchParams.get("derivation_id");
  return value?.trim() || null;
}

async function loadContext(id: string, requestedDerivation: string | null) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return { error: Response.json({ error: "Authentication required" }, { status: 401 }) };

  const { data: membership, error: membershipError } = await supabase
    .from("workspace_members")
    .select("workspace_id")
    .eq("user_id", user.id)
    .eq("status", "active")
    .order("created_at", { ascending: true })
    .limit(1)
    .maybeSingle();
  if (membershipError) throw membershipError;
  if (!membership?.workspace_id) {
    return { error: Response.json({ error: "No active CourtFlux workspace" }, { status: 403 }) };
  }

  const workspaceId = membership.workspace_id as string;
  const admin = createAdminClient();
  const { data: run, error: runError } = await admin
    .from("analysis_runs")
    .select("id,workspace_id,game_id")
    .eq("id", id)
    .eq("workspace_id", workspaceId)
    .maybeSingle();
  if (runError) throw runError;
  if (!run) return { error: Response.json({ error: "Analysis not found" }, { status: 404 }) };
  if (!run.game_id) {
    return {
      error: Response.json({ error: "Analysis is not linked to a Game Record" }, { status: 409 }),
    };
  }

  let derivationQuery = admin
    .from("analysis_derivations")
    .select("id,status")
    .eq("workspace_id", workspaceId)
    .eq("analysis_run_id", run.id)
    .eq("game_id", run.game_id)
    .eq("layer", "p3")
    .eq("derivation_key", "things_to_review");

  if (requestedDerivation) derivationQuery = derivationQuery.eq("id", requestedDerivation);
  else derivationQuery = derivationQuery.eq("status", "current");

  const { data: derivation, error: derivationError } = await derivationQuery
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (derivationError) throw derivationError;
  if (!derivation) {
    return { error: Response.json({ error: "P3 review derivation not found" }, { status: 404 }) };
  }

  return {
    supabase,
    admin,
    userId: user.id,
    workspaceId,
    analysisRunId: run.id as string,
    derivationId: derivation.id as string,
  };
}

async function knownGamePlayers(
  admin: ReturnType<typeof createAdminClient>,
  workspaceId: string,
  analysisRunId: string,
  derivationId: string,
) {
  const { data, error } = await admin
    .from("p3_source_events")
    .select("actor_game_player_id,secondary_game_player_id")
    .eq("workspace_id", workspaceId)
    .eq("analysis_run_id", analysisRunId)
    .eq("derivation_id", derivationId)
    .limit(10000);
  if (error) throw error;
  const values = new Set<string>();
  for (const row of data ?? []) {
    if (typeof row.actor_game_player_id === "string" && row.actor_game_player_id) {
      values.add(row.actor_game_player_id);
    }
    if (typeof row.secondary_game_player_id === "string" && row.secondary_game_player_id) {
      values.add(row.secondary_game_player_id);
    }
  }
  return values;
}

export async function POST(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await context.params;
    const loaded = await loadContext(id, requestedDerivationId(request));
    if ("error" in loaded) return loaded.error;

    let command;
    try {
      command = parseHumanReviewCommand(await request.json());
    } catch (error) {
      return Response.json(
        { error: error instanceof Error ? error.message : String(error) },
        { status: 400 },
      );
    }

    const { data: reviewItem, error: itemError } = await loaded.admin
      .from("p3_review_items")
      .select("id")
      .eq("id", command.review_item_id)
      .eq("workspace_id", loaded.workspaceId)
      .eq("analysis_run_id", loaded.analysisRunId)
      .eq("derivation_id", loaded.derivationId)
      .maybeSingle();
    if (itemError) throw itemError;
    if (!reviewItem) {
      return Response.json(
        { error: "Review item does not belong to the selected P3 derivation" },
        { status: 409 },
      );
    }

    let reviewState: HumanReviewRecord["state"];
    let correctedEventType: HumanReviewRecord["corrected_event_type"] = null;
    let correctedGamePlayerId: string | null = null;

    if (command.type === "confirm") {
      reviewState = "confirmed";
    } else if (command.type === "reject") {
      reviewState = "rejected";
      correctedEventType = "none";
    } else {
      reviewState = "corrected";
      correctedEventType = command.corrected_event_type ?? null;
      correctedGamePlayerId = command.corrected_game_player_id ?? null;
      if (correctedGamePlayerId) {
        const players = await knownGamePlayers(
          loaded.admin,
          loaded.workspaceId,
          loaded.analysisRunId,
          loaded.derivationId,
        );
        if (!players.has(correctedGamePlayerId)) {
          return Response.json(
            { error: "Corrected player is not a known game-player in this P3 derivation" },
            { status: 400 },
          );
        }
      }
    }

    const { data, error } = await loaded.supabase
      .from("p3_reviews")
      .upsert(
        {
          workspace_id: loaded.workspaceId,
          review_item_id: command.review_item_id,
          reviewer_id: loaded.userId,
          review_state: reviewState,
          corrected_event_type: correctedEventType,
          corrected_game_player_id: correctedGamePlayerId,
          note: command.note ?? null,
        },
        { onConflict: "review_item_id,reviewer_id" },
      )
      .select("review_state,corrected_event_type,corrected_game_player_id,note,updated_at")
      .single();

    if (error) {
      if (error.code === "42501") {
        return Response.json(
          { error: "Review permission is required for this workspace" },
          { status: 403 },
        );
      }
      throw error;
    }

    return Response.json(
      {
        review_item_id: command.review_item_id,
        my_review: {
          state: data.review_state,
          corrected_event_type: data.corrected_event_type,
          corrected_game_player_id: data.corrected_game_player_id,
          note: data.note,
          updated_at: data.updated_at,
        } satisfies HumanReviewRecord,
      },
      { headers: { "Cache-Control": "private, no-store" } },
    );
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
