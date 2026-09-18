const ENABLED_VALUES = new Set(["1", "true", "yes", "on"]);

/**
 * Server-side gate for the integrated P3 foundation surfaces.
 *
 * The flag is intentionally fail-closed: an unset or unrecognized value keeps
 * P3 hidden/inactive. Enable it only in environments where the P3 migrations
 * have been applied and the product surface is intentionally being exercised.
 */
export function isP3FoundationEnabled(
  value: string | undefined = process.env.P3_FOUNDATION_ENABLED,
) {
  return ENABLED_VALUES.has((value ?? "").trim().toLowerCase());
}
