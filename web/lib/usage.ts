export const PLAN_MONTHLY_MINUTES: Record<string, number> = {
  beta: 60,
  coach: 300,
  team: 720,
  academy: 2400,
};

export const USAGE_WARNING_PERCENT = 80;
export const USAGE_HARD_STOP_PERCENT = 100;

const PLAN_ALIASES: Record<string, string> = {
  free: "beta",
  base: "coach",
  premium: "team",
};

export type UsageLevel = "healthy" | "warning" | "exhausted";

export function normalizePlanCode(value: string | null | undefined) {
  const code = (value || "beta").toLowerCase();
  return PLAN_ALIASES[code] ?? code;
}

function asFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

export function resolveMonthlyAllowance(
  planCode: string | null | undefined,
  limits: Record<string, unknown> | null | undefined,
  grantedMinutes = 0,
) {
  const normalizedPlan = normalizePlanCode(planCode);
  const configured =
    asFiniteNumber(limits?.analysis_minutes_monthly) ??
    asFiniteNumber(limits?.analysis_minutes);
  const baseAllowance = configured ?? PLAN_MONTHLY_MINUTES[normalizedPlan] ?? null;

  if (baseAllowance === null) return null;
  return Math.max(0, baseAllowance + Math.max(0, grantedMinutes));
}

export function getUsageLevel(usedMinutes: number, allowance: number | null): UsageLevel {
  if (!allowance || allowance <= 0) return "healthy";
  const percent = (Math.max(0, usedMinutes) / allowance) * 100;
  if (percent >= USAGE_HARD_STOP_PERCENT) return "exhausted";
  if (percent >= USAGE_WARNING_PERCENT) return "warning";
  return "healthy";
}

export function getUsagePercent(usedMinutes: number, allowance: number | null) {
  if (!allowance || allowance <= 0) return 0;
  return Math.min(100, Math.max(0, (usedMinutes / allowance) * 100));
}

export function canStartAnalysis(usedMinutes: number, allowance: number | null) {
  return getUsageLevel(usedMinutes, allowance) !== "exhausted";
}
