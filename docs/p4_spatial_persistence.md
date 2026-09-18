# P4 Spatial Persistence Model

**Status:** complete for P4 Spatial Foundation task 2  
**Source contract:** `p4.spatial_frame.v1`  
**Dependency boundary:** platform base + P1/P4 only; no P2 or P3 tables are referenced.

## Purpose

Task 1 froze the in-memory/serialized `SpatialFrame` contract. Task 2 gives that contract durable lineage without making P4 depend on P2 basketball semantics or the P3 Game Record.

The persistence chain is:

```text
analysis_runs
      ↓
p4_spatial_runs
      ├── p4_spatial_snapshots
      └── p4_spatial_artifacts ──→ analysis_artifacts
```

`analysis_runs` and `analysis_artifacts` are platform-level records that pre-date P3. P4 does not use `analysis_derivations`, `p3_source_events`, review items, clips or possession records.

## Hybrid storage policy

A full spatial timeline can contain many frames and player observations. The database should not be forced to store every 25/30 FPS observation as a normalized row merely because the source video is dense.

The persistence model therefore separates:

### Queryable Postgres state

Postgres stores:

- spatial run/version lineage;
- status and provenance;
- sparse or selected `SpatialFrame` snapshots;
- artifact references and time/frame ranges;
- summary counts and diagnostics.

### Dense artifacts

Full-resolution timelines should normally be written as an `analysis_artifact` (for example JSONL/Parquet in a later materialization task) and linked through `p4_spatial_artifacts`.

This keeps the foundation queryable without making database row volume proportional to video FPS × player count.

## `p4_spatial_runs`

One row represents one deterministic P4 spatial materialization configuration over one base `analysis_run`.

Important fields:

```text
id
workspace_id
game_id
analysis_run_id

contract_version
spatial_runtime_version
configuration_version
fingerprint

source_versions
configuration
provenance
output_summary

status
supersedes_id
error
started_at
completed_at
```

### Fingerprint

The application fingerprint includes:

- base `analysis_run_id`;
- optional `game_id`;
- `p4.spatial_frame.v1`;
- P4 spatial runtime version;
- configuration version;
- P1/source versions;
- materialization configuration.

The run id is a deterministic UUID derived from the base run + fingerprint. Repeating the same request therefore resolves to the same spatial run instead of creating duplicate truth.

### Lifecycle

```text
building
   ↓
current
   ↓
superseded

building
   ↓
failed
```

Only one spatial run may be `current` for an `analysis_run`.

`complete_p4_spatial_run(...)` performs the current/superseded transition under a Postgres advisory transaction lock. Historical spatial runs remain addressable.

## `p4_spatial_snapshots`

Snapshots are exact serialized `SpatialFrame` payloads plus a small index:

```text
frame_id
timestamp_ms
snapshot_kind
contract_version
source_runtime_version
geometry_valid
court_visible
player_count
payload
```

Database triggers fail closed when the denormalized index disagrees with the payload, including:

- frame id;
- source timestamp;
- contract version;
- player count;
- geometry validity;
- court visibility;
- source runtime version.

Task 2 creates the persistence boundary only. Which frames are materialized, sampling policies and radar-oriented snapshot strategy belong to later Spatial Foundation tasks.

## `p4_spatial_artifacts`

This table links a spatial run to an existing platform `analysis_artifact`.

It records:

```text
artifact_kind
content_contract_version
content_format
frame range
timestamp range
record count
player-observation count
metadata
```

A trigger requires the linked artifact and P4 spatial run to belong to the same base `analysis_run`.

## Parent-lineage invariants

The database validates rather than repairs lineage:

```text
p4_spatial_run.workspace/game
        =
analysis_run.workspace/game
```

and child rows must match their parent spatial run.

Mismatched lineage fails the write. It is never silently reassigned.

## Security model

All three tables are in `public`, so RLS is enabled explicitly.

Authenticated users receive:

```text
SELECT
```

only when:

```text
private.is_workspace_member(workspace_id)
```

Derived spatial truth is worker-owned. `INSERT`, `UPDATE` and `DELETE` are not granted to authenticated product clients.

`service_role` receives the required CRUD privileges for trusted worker/server materialization.

The lifecycle RPC functions are `SECURITY INVOKER`, not `SECURITY DEFINER`, and execution is revoked from `PUBLIC`, `anon` and `authenticated`; only `service_role` may call them.

The migration includes explicit grants because Supabase no longer guarantees that new `public` tables are automatically reachable through the Data API.

## Dependency boundary

Allowed persistence parents:

```text
workspaces
games
analysis_runs
analysis_artifacts
```

Forbidden dependencies:

```text
P2 events
possessions
game-player identity
analysis_derivations
p3_source_events
p3_review_items
p3_clips
p3_reviews
tactical actions
```

This means the spatial layer can be produced, persisted, inspected and reprocessed while P2 and the final P3 product remain under validation.

## Production status

The migration is source-controlled as part of the P4 Spatial Foundation but is **not applied to the production Supabase project by this task**.

Production rollout remains a separate explicit action.
