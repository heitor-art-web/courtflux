export const ONBOARDING_ROLES = [
  "coach",
  "analyst",
  "team_staff",
  "academy_staff",
  "other",
] as const;

export type OnboardingRole = (typeof ONBOARDING_ROLES)[number];

export const ONBOARDING_ROLE_LABELS: Record<OnboardingRole, string> = {
  coach: "Coach",
  analyst: "Performance analyst",
  team_staff: "Team / club staff",
  academy_staff: "Academy / programme staff",
  other: "Other basketball staff",
};

export function isOnboardingRole(value: unknown): value is OnboardingRole {
  return typeof value === "string" && (ONBOARDING_ROLES as readonly string[]).includes(value);
}
