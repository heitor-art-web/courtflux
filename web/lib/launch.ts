const ENABLED_VALUES = new Set(["1", "true", "yes", "on"]);

export function isPublicLaunchEnabled() {
  return ENABLED_VALUES.has(
    (process.env.COURTFLUX_PUBLIC_LAUNCH_ENABLED ?? "").trim().toLowerCase(),
  );
}
