# P4 Radar / Court Viewer Foundation

**Status:** complete for P4 Spatial Foundation task 13  
**Surface:** internal diagnostic only  
**Route:** `/analysis/[id]/spatial`  
**Feature flag:** `P4_SPATIAL_DIAGNOSTICS_ENABLED`  
**Dependency boundary:** P1/P4 spatial state only; no P2/P3 basketball semantics.

## Purpose

The Radar / Court Viewer is a developer and QA surface for inspecting `p4.spatial_snapshot.v1` on the canonical 28 × 15 m FIBA court before tactical semantics are introduced.

It is intentionally not a customer-facing tactical review experience yet. Its job is to make the spatial foundation observable enough that geometry, smoothing, motion and abstention problems can be diagnosed frame-by-frame.

## Data path

```text
analysis_runs
    ↓
p4_spatial_runs
    ↓
p4_spatial_snapshots
    ↓
p4.spatial_snapshot.v1
    ↓
internal Spatial Radar / Court Viewer
```

The server page reads through the authenticated Supabase client, so existing RLS remains the authorization boundary. There is no service-role key in the browser.

The route is additionally fail-closed behind:

```bash
P4_SPATIAL_DIAGNOSTICS_ENABLED=true
```

Accepted truthy values are `1`, `true`, `yes` and `on`, case-insensitive. When the flag is disabled, the route redirects back to the normal analysis page.

The branch does not configure this environment variable in Vercel.

## Viewer capabilities

The first diagnostic viewer supports:

- canonical FIBA 28 × 15 m court rendering;
- frame id and source timestamp;
- play / pause / scrub timeline controls;
- 0.5×, 1×, 2× and 4× diagnostic playback;
- P1 local-team colouring only;
- raw metric court positions;
- smoothed trajectory positions;
- recent trajectory trails;
- velocity vectors;
- per-track selection and inspection;
- geometry-quality state and reason;
- homography source and age;
- anchor/inlier counts;
- reprojection RMSE;
- projection uncertainty when available;
- spatial eligibility state/reasons;
- metric-player coverage;
- explicit warning when persisted rows fail the web parser.

The UI consumes only a diagnostic subset of the frozen snapshot payload. Relationships, team-shape and generic motion data remain persisted in `p4.spatial_snapshot.v1` but do not need to be visually promoted in this first viewer foundation.

## Coordinate semantics

Court rendering uses the same canonical physical coordinate system as the backend:

```text
p4.fiba_28x15_xy.v1
x: 0 → 28 m
 y: 0 → 15 m
```

The court is shown in physical orientation only. The two baskets are physical left/right baskets.

The viewer does not infer or display:

- attacking direction;
- offensive basket;
- offense or defense roles;
- ball handler;
- possession;
- defender or matchup;
- screen/cut/drive/P&R;
- tactical actions.

Those concepts remain gated until validated P2/P3 basketball context is combined with the spatial foundation.

## Raw versus smoothed evidence

Raw and smoothed positions are deliberately separate overlays.

- `court_position_m` is shown as raw metric evidence.
- `smoothed_position_m` is shown as the main track marker.
- trails prefer smoothed positions and fall back to raw only when needed for diagnostic continuity.
- velocity vectors are drawn from the current derived kinematics state.

This makes smoothing drift, jitter and gap behaviour inspectable rather than hiding them behind one final marker.

## Eligibility and abstention

The viewer preserves the task-6/task-12 semantics:

```text
ELIGIBLE  → metric state can be inspected normally
DEGRADED  → state is visibly marked as degraded
INVALID   → persisted raw evidence may still be inspected, but derived metric truth must be absent
```

The viewer never fills missing spatial truth itself.

If the P4 persistence migration has not been applied in the current environment, the route shows a diagnostic `Spatial data unavailable` state instead of crashing or fabricating sample data.

## Snapshot parsing

The web surface does not cast arbitrary JSON directly into UI state. `web/lib/p4/spatial-viewer.ts` validates the subset needed by the viewer and rejects malformed rows.

The parser requires:

- `p4.spatial_snapshot.v1`;
- non-negative frame/time identifiers;
- valid geometry and court-visible booleans;
- geometry-quality state/reason;
- eligibility state and metric coverage;
- valid player rows with non-negative track ids and finite coordinates.

Rejected persisted rows are counted and surfaced to the operator.

## Scope boundary

This foundation is intentionally diagnostic and reusable. A later P4 Tactical Review experience can reuse the court renderer/playback structure while layering validated basketball/tactical context on top.

That later UX should not require rebuilding:

- court coordinates;
- playback/scrubbing;
- player plotting;
- trajectory trails;
- geometry diagnostics;
- eligibility/abstention display.

## Production status

No Vercel deployment or branch environment configuration is part of this task.

The P4 persistence migration remains source-controlled and not applied to production by this task. The internal route therefore stays harmless in environments where spatial persistence is absent, and the feature flag defaults to disabled.
