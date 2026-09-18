import { normalizeTagName, parseReviewActionCommand } from "@/lib/p3/review-actions";
import type { P3ReviewActionsState } from "@/lib/p3/review-actions";
import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

type Context = {
  userId: string;
  workspaceId: string;
  analysisRunId: string;
  gameId: string;
  derivationId: string | null;
  reviewItemIds: string[];
};

async function loadContext(id: string) {
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

  const { data: derivation, error: derivationError } = await admin
    .from("analysis_derivations")
    .select("id")
    .eq("workspace_id", workspaceId)
    .eq("analysis_run_id", run.id)
    .eq("game_id", run.game_id)
    .eq("layer", "p3")
    .eq("derivation_key", "things_to_review")
    .eq("status", "current")
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (derivationError) throw derivationError;

  let reviewItemIds: string[] = [];
  if (derivation?.id) {
    const { data: items, error: itemError } = await admin
      .from("p3_review_items")
      .select("id")
      .eq("workspace_id", workspaceId)
      .eq("analysis_run_id", run.id)
      .eq("derivation_id", derivation.id);
    if (itemError) throw itemError;
    reviewItemIds = (items ?? []).map((row) => row.id as string);
  }

  return {
    admin,
    context: {
      userId: user.id,
      workspaceId,
      analysisRunId: run.id as string,
      gameId: run.game_id as string,
      derivationId: (derivation?.id as string | undefined) ?? null,
      reviewItemIds,
    } satisfies Context,
  };
}

async function readState(admin: ReturnType<typeof createAdminClient>, context: Context) {
  const { workspaceId, userId, reviewItemIds } = context;
  const [savedResult, notesResult, tagsResult, collectionsResult, tagLinksResult, collectionItemsResult] =
    await Promise.all([
      admin
        .from("p3_saved_items")
        .select("review_item_id")
        .eq("workspace_id", workspaceId)
        .eq("user_id", userId),
      admin
        .from("p3_notes")
        .select("review_item_id,body,updated_at")
        .eq("workspace_id", workspaceId)
        .eq("user_id", userId),
      admin
        .from("p3_tags")
        .select("id,name")
        .eq("workspace_id", workspaceId)
        .eq("user_id", userId)
        .order("normalized_name", { ascending: true }),
      admin
        .from("p3_collections")
        .select("id,name,description")
        .eq("workspace_id", workspaceId)
        .eq("user_id", userId)
        .order("updated_at", { ascending: false }),
      admin
        .from("p3_review_item_tags")
        .select("review_item_id,tag_id")
        .eq("workspace_id", workspaceId)
        .eq("user_id", userId),
      admin
        .from("p3_collection_items")
        .select("collection_id,review_item_id")
        .eq("workspace_id", workspaceId)
        .eq("user_id", userId),
    ]);

  for (const result of [
    savedResult,
    notesResult,
    tagsResult,
    collectionsResult,
    tagLinksResult,
    collectionItemsResult,
  ]) {
    if (result.error) throw result.error;
  }

  const allowed = new Set(reviewItemIds);
  const savedReviewItemIds = (savedResult.data ?? [])
    .map((row) => row.review_item_id as string)
    .filter((id) => allowed.has(id));

  const notesByReviewItem: P3ReviewActionsState["notes_by_review_item"] = {};
  for (const row of notesResult.data ?? []) {
    const id = row.review_item_id as string;
    if (!allowed.has(id)) continue;
    notesByReviewItem[id] = {
      body: row.body as string,
      updated_at: row.updated_at as string,
    };
  }

  const tagIdsByReviewItem: Record<string, string[]> = {};
  for (const row of tagLinksResult.data ?? []) {
    const id = row.review_item_id as string;
    if (!allowed.has(id)) continue;
    (tagIdsByReviewItem[id] ??= []).push(row.tag_id as string);
  }

  const collectionCounts = new Map<string, number>();
  const collectionIdsByReviewItem: Record<string, string[]> = {};
  for (const row of collectionItemsResult.data ?? []) {
    const collectionId = row.collection_id as string;
    collectionCounts.set(collectionId, (collectionCounts.get(collectionId) ?? 0) + 1);
    const reviewItemId = row.review_item_id as string;
    if (!allowed.has(reviewItemId)) continue;
    (collectionIdsByReviewItem[reviewItemId] ??= []).push(collectionId);
  }

  return {
    analysis_run_id: context.analysisRunId,
    game_id: context.gameId,
    derivation_id: context.derivationId,
    saved_review_item_ids: savedReviewItemIds.sort(),
    notes_by_review_item: notesByReviewItem,
    tags: (tagsResult.data ?? []).map((row) => ({ id: row.id as string, name: row.name as string })),
    tag_ids_by_review_item: tagIdsByReviewItem,
    collections: (collectionsResult.data ?? []).map((row) => ({
      id: row.id as string,
      name: row.name as string,
      description: (row.description as string | null) ?? null,
      item_count: collectionCounts.get(row.id as string) ?? 0,
    })),
    collection_ids_by_review_item: collectionIdsByReviewItem,
  } satisfies P3ReviewActionsState;
}

function requireCurrentReviewItem(context: Context, reviewItemId: string) {
  if (!context.reviewItemIds.includes(reviewItemId)) {
    throw new Error("Review item does not belong to the current P3 derivation");
  }
}

export async function GET(
  _request: Request,
  routeContext: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await routeContext.params;
    const loaded = await loadContext(id);
    if ("error" in loaded) return loaded.error;
    return Response.json(await readState(loaded.admin, loaded.context), {
      headers: { "Cache-Control": "private, no-store" },
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}

export async function POST(
  request: Request,
  routeContext: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await routeContext.params;
    const loaded = await loadContext(id);
    if ("error" in loaded) return loaded.error;
    const { admin, context } = loaded;

    let command;
    try {
      command = parseReviewActionCommand(await request.json());
    } catch (error) {
      return Response.json(
        { error: error instanceof Error ? error.message : String(error) },
        { status: 400 },
      );
    }

    if ("review_item_id" in command) {
      requireCurrentReviewItem(context, command.review_item_id);
    }

    switch (command.type) {
      case "save": {
        const { error } = await admin.from("p3_saved_items").upsert(
          {
            workspace_id: context.workspaceId,
            user_id: context.userId,
            review_item_id: command.review_item_id,
          },
          { onConflict: "user_id,review_item_id", ignoreDuplicates: true },
        );
        if (error) throw error;
        break;
      }
      case "unsave": {
        const { error } = await admin
          .from("p3_saved_items")
          .delete()
          .eq("workspace_id", context.workspaceId)
          .eq("user_id", context.userId)
          .eq("review_item_id", command.review_item_id);
        if (error) throw error;
        break;
      }
      case "set_note": {
        const { error } = await admin.from("p3_notes").upsert(
          {
            workspace_id: context.workspaceId,
            user_id: context.userId,
            review_item_id: command.review_item_id,
            body: command.body,
          },
          { onConflict: "user_id,review_item_id" },
        );
        if (error) throw error;
        break;
      }
      case "delete_note": {
        const { error } = await admin
          .from("p3_notes")
          .delete()
          .eq("workspace_id", context.workspaceId)
          .eq("user_id", context.userId)
          .eq("review_item_id", command.review_item_id);
        if (error) throw error;
        break;
      }
      case "create_tag": {
        const normalizedName = normalizeTagName(command.name);
        const { error } = await admin.from("p3_tags").upsert(
          {
            workspace_id: context.workspaceId,
            user_id: context.userId,
            name: command.name,
            normalized_name: normalizedName,
          },
          { onConflict: "workspace_id,user_id,normalized_name" },
        );
        if (error) throw error;
        break;
      }
      case "tag":
      case "untag": {
        const { data: tag, error: tagError } = await admin
          .from("p3_tags")
          .select("id")
          .eq("id", command.tag_id)
          .eq("workspace_id", context.workspaceId)
          .eq("user_id", context.userId)
          .maybeSingle();
        if (tagError) throw tagError;
        if (!tag) return Response.json({ error: "Tag not found" }, { status: 404 });

        if (command.type === "tag") {
          const { error } = await admin.from("p3_review_item_tags").upsert(
            {
              workspace_id: context.workspaceId,
              user_id: context.userId,
              review_item_id: command.review_item_id,
              tag_id: command.tag_id,
            },
            { onConflict: "review_item_id,tag_id,user_id", ignoreDuplicates: true },
          );
          if (error) throw error;
        } else {
          const { error } = await admin
            .from("p3_review_item_tags")
            .delete()
            .eq("workspace_id", context.workspaceId)
            .eq("user_id", context.userId)
            .eq("review_item_id", command.review_item_id)
            .eq("tag_id", command.tag_id);
          if (error) throw error;
        }
        break;
      }
      case "create_collection": {
        const { error } = await admin.from("p3_collections").insert({
          workspace_id: context.workspaceId,
          user_id: context.userId,
          name: command.name,
          description: command.description ?? null,
        });
        if (error) throw error;
        break;
      }
      case "add_to_collection":
      case "remove_from_collection": {
        const { data: collection, error: collectionError } = await admin
          .from("p3_collections")
          .select("id")
          .eq("id", command.collection_id)
          .eq("workspace_id", context.workspaceId)
          .eq("user_id", context.userId)
          .maybeSingle();
        if (collectionError) throw collectionError;
        if (!collection) return Response.json({ error: "Collection not found" }, { status: 404 });

        if (command.type === "add_to_collection") {
          const { error } = await admin.from("p3_collection_items").upsert(
            {
              workspace_id: context.workspaceId,
              user_id: context.userId,
              collection_id: command.collection_id,
              review_item_id: command.review_item_id,
            },
            { onConflict: "collection_id,review_item_id", ignoreDuplicates: true },
          );
          if (error) throw error;
        } else {
          const { error } = await admin
            .from("p3_collection_items")
            .delete()
            .eq("workspace_id", context.workspaceId)
            .eq("user_id", context.userId)
            .eq("collection_id", command.collection_id)
            .eq("review_item_id", command.review_item_id);
          if (error) throw error;
        }
        break;
      }
    }

    return Response.json(await readState(admin, context), {
      headers: { "Cache-Control": "private, no-store" },
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
