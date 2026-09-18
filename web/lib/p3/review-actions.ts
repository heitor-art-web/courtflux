export type P3ReviewActionCommand =
  | { type: "save"; review_item_id: string }
  | { type: "unsave"; review_item_id: string }
  | { type: "set_note"; review_item_id: string; body: string }
  | { type: "delete_note"; review_item_id: string }
  | { type: "create_tag"; name: string }
  | { type: "tag"; review_item_id: string; tag_id: string }
  | { type: "untag"; review_item_id: string; tag_id: string }
  | { type: "create_collection"; name: string; description?: string | null }
  | { type: "add_to_collection"; review_item_id: string; collection_id: string }
  | { type: "remove_from_collection"; review_item_id: string; collection_id: string };

export type P3ReviewActionsState = {
  analysis_run_id: string;
  game_id: string;
  derivation_id: string | null;
  saved_review_item_ids: string[];
  notes_by_review_item: Record<string, { body: string; updated_at: string }>;
  tags: Array<{ id: string; name: string }>;
  tag_ids_by_review_item: Record<string, string[]>;
  collections: Array<{ id: string; name: string; description: string | null; item_count: number }>;
  collection_ids_by_review_item: Record<string, string[]>;
};

export function normalizeTagName(value: string) {
  return value.trim().replace(/\s+/g, " ").toLocaleLowerCase("en");
}

function requireId(value: unknown, field: string) {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${field} is required`);
  }
  return value.trim();
}

function requireText(value: unknown, field: string, max: number) {
  if (typeof value !== "string") throw new Error(`${field} is required`);
  const text = value.trim();
  if (!text) throw new Error(`${field} is required`);
  if (text.length > max) throw new Error(`${field} must be ${max} characters or fewer`);
  return text;
}

export function parseReviewActionCommand(value: unknown): P3ReviewActionCommand {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("A review action command object is required");
  }
  const raw = value as Record<string, unknown>;
  const type = requireId(raw.type, "type") as P3ReviewActionCommand["type"];

  switch (type) {
    case "save":
    case "unsave":
    case "delete_note":
      return { type, review_item_id: requireId(raw.review_item_id, "review_item_id") };
    case "set_note":
      return {
        type,
        review_item_id: requireId(raw.review_item_id, "review_item_id"),
        body: requireText(raw.body, "body", 4000),
      };
    case "create_tag": {
      const name = requireText(raw.name, "name", 64);
      return { type, name };
    }
    case "tag":
    case "untag":
      return {
        type,
        review_item_id: requireId(raw.review_item_id, "review_item_id"),
        tag_id: requireId(raw.tag_id, "tag_id"),
      };
    case "create_collection":
      return {
        type,
        name: requireText(raw.name, "name", 120),
        description:
          raw.description == null || raw.description === ""
            ? null
            : requireText(raw.description, "description", 1000),
      };
    case "add_to_collection":
    case "remove_from_collection":
      return {
        type,
        review_item_id: requireId(raw.review_item_id, "review_item_id"),
        collection_id: requireId(raw.collection_id, "collection_id"),
      };
    default:
      throw new Error(`Unsupported review action: ${String(type)}`);
  }
}
