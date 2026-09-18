const TRUE_VALUES = new Set(["1", "true", "yes", "on"]);

export function isP4SpatialDiagnosticsEnabled() {
  const value = process.env.P4_SPATIAL_DIAGNOSTICS_ENABLED?.trim().toLowerCase();
  return value ? TRUE_VALUES.has(value) : false;
}
