# P4 Spatial Persistence Model

**Status:** complete for P4 Spatial Foundation task 2; snapshot payload upgraded by task 12  
**Source contract:** `p4.spatial_frame.v1`  
**Materialized snapshot contract:** `p4.spatial_snapshot.v1`  
**Dependency boundary:** platform base + P1/P4 only; no P2 or P3 tables are referenced.

## Purpose

Task 1 froze the P1→P4 `SpatialFrame` contract. Task 2 gives P4 durable lineage without making it depend on P2 basketball semantics or the P3 Game Record.

Task 12 adds materialized spatial state without changing that source boundary:

```text
analysis_runs
      ↓
p4_spatial_runs                     contract = p4.spatial_frame.v1
      ├── p4_spatial_snapshots      contract = p4.spatial_snapshot.v1
      └── p4_spatial_artifacts ──→ analysis_artifacts
```

`analysis_runs` and `analysis_artifacts` are platform-level records that pre-date P3. P4 does not use `analysis_derivations`, P3 review/clip records or possession records.

## Hybrid storage policy

Postgres stores run/version lineage, status/provenance, sparse or selected materialized snapshots, artifact references and summary/diagnostic data. Dense timelines may live as `analysis_artifacts` linked through `p4_spatial_artifacts`.

Supported artifact contracts are explicit:

```text
p4.spatial_frame.v1
p4.spatial_snapshot.v1
```

## `p4_spatial_runs`

One row represents one deterministic P4 spatial materialization configuration over one base `analysis_run`.

The run remains rooted in:

```text
p4.spatial_frame.v1
```

because that is the frozen upstream P1→P4 source boundary.

The fingerprint includes the base analysis run, optional game, source contract, P4 runtime/config versions, source versions and configuration. Repeating the same request therefore resolves to the same run identity.

Lifecycle remains:

```text
building → current → superseded
building → failed
```

Only one run may be current per `analysis_run`.

## `p4_spatial_snapshots`

The table now stores exact serialized `p4.spatial_snapshot.v1` payloads plus a small query index:

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

Each materialized snapshot declares:

```text
source_spatial_contract_version = p4.spatial_frame.v1
```

The database validates rather than repairs the denormalized index. It checks payload/index contract, source contract vs parent run, frame/timestamp, geometry-quality and eligibility alignment, player count, geometry validity, court visibility, source runtime version and array shape of derived collections.

For `INVALID` eligibility, persisted relationships, team shapes and motion primitives must be empty. The Python contract additionally prevents derived per-player smoothed/velocity/speed truth on invalid frames.

## `p4_spatial_artifacts`

Artifacts can explicitly carry either raw `p4.spatial_frame.v1` or materialized `p4.spatial_snapshot.v1` content. A trigger still requires artifact and spatial run to belong to the same base `analysis_run`.

## Parent-lineage invariants

```text
p4_spatial_run.workspace/game
        =
analysis_run.workspace/game
```

Child rows must match the parent run. Snapshot and parent run have different contracts by design, so the snapshot's `source_spatial_contract_version` must equal the parent run contract.

## Security model

All three tables are in `public` with RLS enabled. Authenticated users receive `SELECT` only when `private.is_workspace_member(workspace_id)` succeeds. Derived writes remain service-owned.

Lifecycle RPCs remain `SECURITY INVOKER`; execution is revoked from `PUBLIC`, `anon` and `authenticated`, with `service_role` as the trusted caller.

Explicit grants remain intentional because Supabase is moving public-schema Data API exposure to an opt-in grant model.

## Dependency boundary

Allowed parents:

```text
workspaces
games
analysis_runs
analysis_artifacts
```

Forbidden dependencies include P2 events, possessions, game-player identity, P3 derivations/review/clips and tactical actions.

## Production status

The migration remains source-controlled on `p4_spatial_foundation` and is **not applied to production by this task**.

Because the P4 schema has not yet been deployed to production, task 12 updates the branch migration in place so the eventual first deployment creates the final foundation snapshot schema directly.
