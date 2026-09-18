export type P3ReviewActionCommand =
  | { type: "save"; review_item_id: string }
  | { type: "unsave"; review_item_id: string }
  | { type: "set_note"; review_item_id: string; body: string }
  | { type: "delete_note"; review_item_id: string }
  | { type: "create_tag"; name: string }
  | { type: "rename_tag"; tag_id: string; name: string }
  | { type: "delete_tag"; tag_id: string }
  | { type: "tag"; review_item_id: string; tag_id: string }
  | { type: "untag"; review_item_id: string; tag_id: string }
  | { type: "create_collection"; name: string; description?: string | null }
  | { type: "update_collection"; collection_id: string; name: string; description?: string | null }
  | { type: "delete_collection"; collection_id: string }
  | { type: "add_to_collection"; review_item_id: string; collection_id: string }
  | { type: "remove_from_collection"; review_item_id: string; collection_id: string };

export type P3ReviewActionsState = {
  analysis_run_id: string;
  game_id: string;
  derivation_id: string | null;
  derivation_status: string | null;
  available_derivations: Array<{
    id: string;
    status: string;
    generated_at: string;
  }>;
  saved_review_item_ids: string[];
  saved_items: Array<{ review_item_id: string; created_at: string }>;
  notes_by_review_item: Record<
    string,
    { body: string; created_at: string; updated_at: string }
  >;
  tags: Array<{
    id: string;
    name: string;
    created_at: string;
    updated_at: string;
    item_count: number;
    total_item_count: number;
  }>;
  tag_ids_by_review_item: Record<string, string[]>;
  collections: Array<{
    id: string;
    name: string;
    description: string | null;
    created_at: string;
    updated_at: string;
    item_count: number;
    total_item_count: number;
  }>;
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

function optionalText(value: unknown, field: string, max: number) {
  if (value == null || value === "") return null;
  return requireText(value, field, max);
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
    case "create_tag":
      return { type, name: requireText(raw.name, "name", 64) };
    case "rename_tag":
      return {
        type,
        tag_id: requireId(raw.tag_id, "tag_id"),
        name: requireText(raw.name, "name", 64),
      };
    case "delete_tag":
      return { type, tag_id: requireId(raw.tag_id, "tag_id") };
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
        description: optionalText(raw.description, "description", 1000),
      };
    case "update_collection":
      return {
        type,
        collection_id: requireId(raw.collection_id, "collection_id"),
        name: requireText(raw.name, "name", 120),
        description: optionalText(raw.description, "description", 1000),
      };
    case "delete_collection":
      return { type, collection_id: requireId(raw.collection_id, "collection_id") };
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
