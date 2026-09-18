export const HUMAN_REVIEW_EVENT_TYPES = [
  "pass",
  "turnover",
  "shot",
  "rebound",
  "loose_ball",
  "dunk",
  "alley_oop",
] as const;

export type HumanReviewEventType = (typeof HUMAN_REVIEW_EVENT_TYPES)[number];

export type HumanReviewRecord = {
  state: "confirmed" | "corrected" | "rejected";
  corrected_event_type: HumanReviewEventType | "none" | null;
  corrected_game_player_id: string | null;
  note: string | null;
  updated_at: string;
};

export type HumanReviewCommand =
  | { type: "confirm"; review_item_id: string; note?: string | null }
  | {
      type: "correct";
      review_item_id: string;
      corrected_event_type?: HumanReviewEventType | null;
      corrected_game_player_id?: string | null;
      note?: string | null;
    }
  | { type: "reject"; review_item_id: string; note?: string | null };

function requireId(value: unknown, field: string) {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${field} is required`);
  return value.trim();
}

function optionalText(value: unknown, field: string, max: number) {
  if (value == null || value === "") return null;
  if (typeof value !== "string") throw new Error(`${field} must be text`);
  const text = value.trim();
  if (!text) return null;
  if (text.length > max) throw new Error(`${field} must be ${max} characters or fewer`);
  return text;
}

function optionalEventType(value: unknown) {
  if (value == null || value === "") return null;
  if (typeof value !== "string" || !HUMAN_REVIEW_EVENT_TYPES.includes(value as HumanReviewEventType)) {
    throw new Error("corrected_event_type is not supported");
  }
  return value as HumanReviewEventType;
}

export function parseHumanReviewCommand(value: unknown): HumanReviewCommand {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("A human review command object is required");
  }
  const raw = value as Record<string, unknown>;
  const type = requireId(raw.type, "type");
  const reviewItemId = requireId(raw.review_item_id, "review_item_id");
  const note = optionalText(raw.note, "note", 2000);

  if (type === "confirm") return { type, review_item_id: reviewItemId, note };
  if (type === "reject") return { type, review_item_id: reviewItemId, note };
  if (type === "correct") {
    const correctedEventType = optionalEventType(raw.corrected_event_type);
    const correctedGamePlayerId = optionalText(
      raw.corrected_game_player_id,
      "corrected_game_player_id",
      200,
    );
    if (!correctedEventType && !correctedGamePlayerId) {
      throw new Error("A corrected event or corrected player is required");
    }
    return {
      type,
      review_item_id: reviewItemId,
      corrected_event_type: correctedEventType,
      corrected_game_player_id: correctedGamePlayerId,
      note,
    };
  }

  throw new Error(`Unsupported human review action: ${type}`);
}
