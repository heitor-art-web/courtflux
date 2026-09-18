# P3.0 — Analytics Contract v1

**Status:** frozen for P3 foundation development.

P3.0 defines the versioned boundary between P2 basketball understanding and P3 product analytics/review. It freezes the **shape and interpretation of data crossing the boundary**; it does not claim that P2 event accuracy has passed the P2 exit gate.

The operational rule remains:

```text
P2 basketball truth changes
        ↓
new source/runtime/config version
        ↓
P3 derivation becomes reprocessable/stale
        ↓
re-materialize P3

not

rewrite the P3 product architecture
```

## Canonical normalized event

The current P3 contract version is:

```text
p3.source_event.v1
```

Every normalized source event contains the following fields:

```text
source_event_id
game_id
analysis_run_id

event_type
decision
confidence
uncertainty

frame_start
frame_end
start_ms
end_ms

actor_track_id
secondary_track_id
actor_game_player_id
secondary_game_player_id
team_id
location_court_m

evidence
attributes
provenance

source_contract_version
source_runtime_version
contract_version
```

`start_ms` / `end_ms` are **source-video milliseconds**. They are the canonical evidence timebase used to recreate physical clips. Game clock may be added later as display/context data but cannot replace source-video time.

## Explicit unknowns

Unknown values are represented explicitly as `null` / `None` for nullable scalar fields. The adapter must not fabricate replacement values.

Examples:

```text
actor_game_player_id = null
team_id = null
start_ms = null
end_ms = null
source_runtime_version = null
```

A missing canonical serialized P2 field is not equivalent to an explicit unknown. Missing fields are rejected at the boundary so producer drift cannot silently become product truth.

## Confidence and support

P3 keeps these concepts separate:

```text
source decision: candidate | confirmed
source confidence: 0..1
source uncertainty: 0..1
human review state: separate P3 record
aggregation coverage: separate derived metric
```

The adapter never promotes a candidate, changes confidence, or converts uncertainty into a generic score.

Invalid probabilities are rejected rather than clamped.

## Current P2 event types accepted by v1

```text
possession_acquired
possession_change
pass_candidate
turnover_candidate
loose_ball_started
loose_ball_recovered
loose_ball_ended
shot_attempt
rebound_candidate
```

The P3 contract test intentionally compares this frozen set with `BasketballEventType`. If P2 adds/removes/renames an event primitive, CI must fail until the P3 boundary is reviewed explicitly.

This is consumer-driven contract testing, not an attempt to freeze P2 accuracy.

## Batch invariants

One P3 source-event batch must have:

- one `game_id`;
- one `analysis_run_id`;
- unique `source_event_id` values;
- one P2 source contract version;
- at most one known P2 runtime version;
- only `p3.source_event.v1` normalized payloads.

An empty batch is valid for a completed run with no emitted P2 events.

## Serialized P2 input

The canonical P2 JSONL row must explicitly contain the fields emitted by `BasketballEvent.as_dict()`:

```text
event_id
event_type
frame_start
frame_end
timestamp_start_sec
timestamp_end_sec
confidence
uncertainty
decision
actor_track_id
secondary_track_id
team_id
location_court_m
evidence
provenance
attributes
```

Nullable values may be `null`, but the canonical field itself must be present.

P3 may enrich the normalized record with game-local player IDs through a resolver. If the resolver cannot resolve a track, the track ID is retained and `game_player_id` remains unknown.

## Versioning policy

The following require an explicit P3 contract review and normally a version bump:

- adding/removing/renaming normalized fields;
- changing field semantics or time units;
- adding/removing/renaming P2 event primitives consumed by P3;
- changing candidate/confirmed meaning;
- changing nullability in a way that changes interpretation.

The following do **not** inherently require a schema version bump, but must remain traceable through source/runtime/config versions and reprocessing:

- P2 model/heuristic improvements;
- confidence calibration changes;
- threshold changes;
- review-policy configuration changes;
- clip-context configuration changes.

## Persistence guardrails

`p3_source_events` mirrors the v1 contract with database constraints for:

- supported event types;
- candidate/confirmed decision values;
- probability ranges;
- frame/time ordering;
- non-empty identifiers/version strings;
- `p3.source_event.v1` contract version.

Browser clients cannot mutate source-event truth. Derived truth remains service-owned; human corrections and review actions live in separate P3 tables.

## P3.0 exit checklist

P3.0 is complete when all of the following hold:

- [x] canonical P2 → P3 event adapter exists;
- [x] source-video milliseconds are the evidence timebase;
- [x] confidence, uncertainty and decision remain separate;
- [x] unknown identity/time values remain explicit;
- [x] original evidence, attributes and provenance are preserved;
- [x] P2 runtime/source contract version is stored;
- [x] P3 normalized contract is versioned;
- [x] invalid source values are rejected instead of repaired;
- [x] duplicate/mixed-run/mixed-version batches are rejected;
- [x] frozen fixture covers every current P2 event type;
- [x] P2 event-enum drift fails contract tests;
- [x] persistence constraints mirror the frozen v1 boundary;
- [x] derivation fingerprints remain deterministic and configuration-sensitive.

P3.0 completion does **not** unblock accuracy-dependent product semantics by itself. Possession truth, final rates, ranking and accuracy-derived insights remain configurable/reprocessable until P2 passes its exit gate.
