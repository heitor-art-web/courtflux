-- P3 foundation: durable normalized copy of the P2 -> P3 source-event contract.
--
-- This table is deliberately source-oriented. It preserves P2 candidate/confirmed
-- semantics and confidence instead of materializing final P3 statistics. A later P2
-- calibration can produce a new derivation while historical rows remain traceable.

create table if not exists public.p3_source_events (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid not null references public.games(id) on delete cascade,
  analysis_run_id uuid not null references public.analysis_runs(id) on delete cascade,
  derivation_id uuid not null references public.analysis_derivations(id) on delete cascade,

  source_event_id text not null,
  event_type text not null,
  decision text not null check (decision in ('candidate','confirmed')),
  confidence double precision not null check (confidence >= 0 and confidence <= 1),
  uncertainty double precision not null check (uncertainty >= 0 and uncertainty <= 1),

  frame_start integer not null check (frame_start >= 0),
  frame_end integer not null check (frame_end >= frame_start),
  start_ms bigint check (start_ms is null or start_ms >= 0),
  end_ms bigint check (end_ms is null or end_ms >= coalesce(start_ms, 0)),

  actor_track_id bigint,
  secondary_track_id bigint,
  actor_game_player_id text,
  secondary_game_player_id text,
  team_id text,
  location_court_m jsonb,

  evidence jsonb not null default '[]'::jsonb,
  attributes jsonb not null default '{}'::jsonb,
  provenance jsonb not null default '{}'::jsonb,

  contract_version text not null,
  source_contract_version text not null,
  source_runtime_version text,

  created_at timestamptz not null default now(),

  unique (derivation_id, source_event_id),
  check (jsonb_typeof(evidence) = 'array'),
  check (jsonb_typeof(attributes) = 'object'),
  check (jsonb_typeof(provenance) = 'object'),
  check (location_court_m is null or jsonb_typeof(location_court_m) = 'array')
);

create index if not exists p3_source_events_game_time_idx
  on public.p3_source_events(game_id, start_ms, frame_start);

create index if not exists p3_source_events_run_type_idx
  on public.p3_source_events(analysis_run_id, event_type, decision);

create index if not exists p3_source_events_actor_idx
  on public.p3_source_events(game_id, actor_game_player_id, event_type)
  where actor_game_player_id is not null;

alter table public.p3_source_events enable row level security;

create policy p3_source_events_select
  on public.p3_source_events
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));

-- Worker/service code owns derived truth. Product clients receive read access and later
-- write corrections into dedicated review tables rather than mutating source events.
revoke all on public.p3_source_events from public, anon, authenticated;
grant select on public.p3_source_events to authenticated;
grant select, insert, update, delete on public.p3_source_events to service_role;

comment on table public.p3_source_events is
  'Durable normalized P2 events consumed by P3. Rows preserve source uncertainty and belong to one versioned analysis_derivation.';
comment on column public.p3_source_events.actor_game_player_id is
  'Optional game-local identity. NULL is valid while P2 game-player attribution remains unresolved; actor_track_id is retained as source evidence.';
