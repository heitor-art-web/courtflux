-- P3.0 frozen source-event contract v1 guardrails.
-- These constraints intentionally make producer drift explicit instead of allowing a
-- new P2 event shape/type to enter P3 persistence silently.

alter table public.p3_source_events
  add constraint p3_source_events_contract_v1_check
  check (contract_version = 'p3.source_event.v1');

alter table public.p3_source_events
  add constraint p3_source_events_event_type_v1_check
  check (
    event_type in (
      'possession_acquired',
      'possession_change',
      'pass_candidate',
      'turnover_candidate',
      'loose_ball_started',
      'loose_ball_recovered',
      'loose_ball_ended',
      'shot_attempt',
      'rebound_candidate'
    )
  );

alter table public.p3_source_events
  add constraint p3_source_events_nonempty_lineage_v1_check
  check (
    btrim(source_event_id) <> ''
    and btrim(event_type) <> ''
    and btrim(source_contract_version) <> ''
    and btrim(contract_version) <> ''
    and (source_runtime_version is null or btrim(source_runtime_version) <> '')
  );

comment on constraint p3_source_events_contract_v1_check on public.p3_source_events is
  'P3.0 source-event boundary is frozen at p3.source_event.v1; schema/semantic changes require an explicit migration/version decision.';

comment on constraint p3_source_events_event_type_v1_check on public.p3_source_events is
  'Consumer-driven P2->P3 event-type guard. New/renamed P2 event primitives require explicit P3 contract review.';
