import type { P3ReviewActionsState } from "@/lib/p3/review-actions";

export type ReviewItem = {
  id: string;
  review_type: string;
  game_player_ids: string[];
  team_id: string | null;
  confidence: number | null;
  support_state: string;
  completeness: string;
  reason_shown: Record<string, unknown>;
  source_time: {
    start_ms: number | null;
    end_ms: number | null;
    frame_start: number | null;
    frame_end: number | null;
  };
  clip: {
    id: string;
    status: string;
    error: string | null;
    clip_start_ms: number;
    clip_end_ms: number;
    artifact: { playable_url: string | null } | null;
  } | null;
  my_review: {
    state: string;
    corrected_event_type: string | null;
    corrected_game_player_id: string | null;
  } | null;
};

export type ThingsToReviewResponse = {
  analysis_run_id: string;
  game_id: string;
  derivation: { id: string; generated_at: string } | null;
  ranking_applied: boolean;
  summary: {
    total_items: number;
    counts_by_type: Record<string, number>;
    counts_by_support_state: Record<string, number>;
    clips: Record<string, number>;
    reviewed_by_me: number;
  };
  items: ReviewItem[];
  error?: string;
};

export type ViewMode =
  | { kind: "all" }
  | { kind: "saved" }
  | { kind: "collection"; collectionId: string };

export type SubmitReviewAction = (
  command: import("@/lib/p3/review-actions").P3ReviewActionCommand,
  key: string,
  successMessage: string,
) => Promise<P3ReviewActionsState | null>;

export function humanize(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatTime(milliseconds: number | null) {
  if (milliseconds == null || !Number.isFinite(milliseconds)) return "No source time";
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`
    : `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export function formatConfidence(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "Unknown confidence";
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}% confidence`;
}

export function reasonLines(reason: Record<string, unknown>) {
  return Object.entries(reason)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .slice(0, 4)
    .map(([key, value]) => {
      const rendered = typeof value === "string" || typeof value === "number" || typeof value === "boolean"
        ? String(value)
        : Array.isArray(value)
          ? value.map(String).join(", ")
          : "available";
      return `${humanize(key)}: ${rendered}`;
    });
}
