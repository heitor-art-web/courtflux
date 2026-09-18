-- P4 Spatial Foundation task 2: durable P1-only spatial persistence.
--
-- Task 12 upgrades snapshot payloads from raw SpatialFrame serialization to the
-- materialized p4.spatial_snapshot.v1 contract while keeping the spatial run
-- rooted in p4.spatial_frame.v1.
--
-- This migration intentionally depends only on platform-level CourtFlux records:
-- workspaces, games, analysis_runs and analysis_artifacts. It does not reference
-- P2 events, P3 derivations/review/clips, possessions or tactical semantics.

create table if not exists public.p4_spatial_runs (
  id uuid primary key,
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid references public.games(id) on delete set null,
  analysis_run_id uuid not null references public.analysis_runs(id) on delete cascade,

  contract_version text not null,
  spatial_runtime_version text not null,
  configuration_version text not null,
  fingerprint text not null,

  source_versions jsonb not null default '{}'::jsonb,
  configuration jsonb not null default '{}'::jsonb,
  provenance jsonb not null default '{}'::jsonb,
  output_summary jsonb not null default '{}'::jsonb,

  status text not null default 'building'
    check (status in ('building','current','superseded','failed')),
  supersedes_id uuid references public.p4_spatial_runs(id) on delete set null,
  error text,

  started_at timestamptz not null default now(),
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  unique (analysis_run_id, fingerprint),

  check (contract_version = 'p4.spatial_frame.v1'),
  check (length(trim(spatial_runtime_version)) > 0),
  check (length(trim(configuration_version)) > 0),
  check (length(trim(fingerprint)) > 0),
  check (jsonb_typeof(source_versions) = 'object'),
  check (jsonb_typeof(configuration) = 'object'),
  check (jsonb_typeof(provenance) = 'object'),
  check (jsonb_typeof(output_summary) = 'object'),
  check (completed_at is null or completed_at >= started_at)
);

create index if not exists p4_spatial_runs_analysis_idx
  on public.p4_spatial_runs(analysis_run_id, created_at desc);

create index if not exists p4_spatial_runs_game_idx
  on public.p4_spatial_runs(game_id, created_at desc)
  where game_id is not null;

create index if not exists p4_spatial_runs_workspace_status_idx
  on public.p4_spatial_runs(workspace_id, status, created_at desc);

create unique index if not exists p4_spatial_runs_one_current_idx
  on public.p4_spatial_runs(analysis_run_id)
  where status = 'current';


create table if not exists public.p4_spatial_snapshots (
  id uuid primary key,
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid references public.games(id) on delete set null,
  analysis_run_id uuid not null references public.analysis_runs(id) on delete cascade,
  spatial_run_id uuid not null references public.p4_spatial_runs(id) on delete cascade,

  frame_id bigint not null check (frame_id >= 0),
  timestamp_ms bigint not null check (timestamp_ms >= 0),
  snapshot_kind text not null default 'materialized',

  contract_version text not null,
  source_runtime_version text,
  geometry_valid boolean not null,
  court_visible boolean not null,
  player_count integer not null check (player_count >= 0),

  payload jsonb not null,
  created_at timestamptz not null default now(),

  unique (spatial_run_id, frame_id),

  check (length(trim(snapshot_kind)) > 0),
  check (contract_version = 'p4.spatial_snapshot.v1'),
  check (source_runtime_version is null or length(trim(source_runtime_version)) > 0),
  check (jsonb_typeof(payload) = 'object')
);

create index if not exists p4_spatial_snapshots_run_time_idx
  on public.p4_spatial_snapshots(spatial_run_id, timestamp_ms, frame_id);

create index if not exists p4_spatial_snapshots_game_time_idx
  on public.p4_spatial_snapshots(game_id, timestamp_ms)
  where game_id is not null;


create table if not exists public.p4_spatial_artifacts (
  id uuid primary key,
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid references public.games(id) on delete set null,
  analysis_run_id uuid not null references public.analysis_runs(id) on delete cascade,
  spatial_run_id uuid not null references public.p4_spatial_runs(id) on delete cascade,
  artifact_id uuid not null references public.analysis_artifacts(id) on delete cascade,

  artifact_kind text not null,
  content_contract_version text not null,
  content_format text,

  frame_start bigint check (frame_start is null or frame_start >= 0),
  frame_end bigint check (frame_end is null or frame_end >= coalesce(frame_start, 0)),
  start_ms bigint check (start_ms is null or start_ms >= 0),
  end_ms bigint check (end_ms is null or end_ms >= coalesce(start_ms, 0)),
  record_count bigint check (record_count is null or record_count >= 0),
  player_observation_count bigint
    check (player_observation_count is null or player_observation_count >= 0),

  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  unique (spatial_run_id, artifact_id, artifact_kind),

  check (length(trim(artifact_kind)) > 0),
  check (
    content_contract_version in (
      'p4.spatial_frame.v1',
      'p4.spatial_snapshot.v1'
    )
  ),
  check (content_format is null or length(trim(content_format)) > 0),
  check (jsonb_typeof(metadata) = 'object')
);

create index if not exists p4_spatial_artifacts_run_idx
  on public.p4_spatial_artifacts(spatial_run_id, artifact_kind, created_at);

create index if not exists p4_spatial_artifacts_artifact_idx
  on public.p4_spatial_artifacts(artifact_id);


create or replace function private.ensure_p4_spatial_run_parent()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  parent_workspace_id uuid;
  parent_game_id uuid;
begin
  select ar.workspace_id, ar.game_id
    into parent_workspace_id, parent_game_id
  from public.analysis_runs ar
  where ar.id = new.analysis_run_id;

  if not found then
    raise exception 'analysis_run_id % does not exist', new.analysis_run_id;
  end if;

  if new.workspace_id is distinct from parent_workspace_id then
    raise exception 'p4_spatial_runs.workspace_id must match analysis_runs.workspace_id';
  end if;

  if new.game_id is distinct from parent_game_id then
    raise exception 'p4_spatial_runs.game_id must match analysis_runs.game_id';
  end if;

  return new;
end;
$$;

revoke all on function private.ensure_p4_spatial_run_parent() from public, anon, authenticated;

create trigger p4_spatial_runs_parent_consistency
  before insert or update of workspace_id, game_id, analysis_run_id
  on public.p4_spatial_runs
  for each row execute function private.ensure_p4_spatial_run_parent();


create or replace function private.ensure_p4_spatial_snapshot_parent()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  parent_workspace_id uuid;
  parent_game_id uuid;
  parent_analysis_run_id uuid;
  parent_contract_version text;
  eligibility_state text;
begin
  select sr.workspace_id, sr.game_id, sr.analysis_run_id, sr.contract_version
    into parent_workspace_id, parent_game_id, parent_analysis_run_id, parent_contract_version
  from public.p4_spatial_runs sr
  where sr.id = new.spatial_run_id;

  if not found then
    raise exception 'spatial_run_id % does not exist', new.spatial_run_id;
  end if;

  if new.workspace_id is distinct from parent_workspace_id
     or new.game_id is distinct from parent_game_id
     or new.analysis_run_id is distinct from parent_analysis_run_id then
    raise exception 'p4_spatial_snapshots lineage must match p4_spatial_runs';
  end if;

  if new.payload ->> 'contract_version' is distinct from new.contract_version then
    raise exception 'snapshot payload contract_version must match indexed contract_version';
  end if;

  if new.payload ->> 'source_spatial_contract_version'
       is distinct from parent_contract_version then
    raise exception 'snapshot source_spatial_contract_version must match p4_spatial_runs';
  end if;

  if (new.payload ->> 'frame_id')::bigint is distinct from new.frame_id then
    raise exception 'snapshot payload frame_id must match indexed frame_id';
  end if;

  if (new.payload ->> 'timestamp_ms')::bigint is distinct from new.timestamp_ms then
    raise exception 'snapshot payload timestamp_ms must match indexed timestamp_ms';
  end if;

  if (new.payload #>> '{geometry_quality,frame_id}')::bigint
       is distinct from new.frame_id
     or (new.payload #>> '{geometry_quality,timestamp_ms}')::bigint
       is distinct from new.timestamp_ms then
    raise exception 'snapshot geometry_quality must match indexed frame/time';
  end if;

  if (new.payload #>> '{spatial_eligibility,frame_id}')::bigint
       is distinct from new.frame_id
     or (new.payload #>> '{spatial_eligibility,timestamp_ms}')::bigint
       is distinct from new.timestamp_ms then
    raise exception 'snapshot spatial_eligibility must match indexed frame/time';
  end if;

  if jsonb_typeof(new.payload -> 'players') is distinct from 'array' then
    raise exception 'snapshot payload players must be an array';
  end if;

  if jsonb_array_length(new.payload -> 'players') is distinct from new.player_count then
    raise exception 'snapshot payload player count must match indexed player_count';
  end if;

  if jsonb_typeof(new.payload -> 'relationships') is distinct from 'array'
     or jsonb_typeof(new.payload -> 'team_shapes') is distinct from 'array'
     or jsonb_typeof(new.payload -> 'motion_primitives') is distinct from 'array' then
    raise exception 'snapshot derived collections must be arrays';
  end if;

  if (new.payload #>> '{geometry,geometry_valid}')::boolean
       is distinct from new.geometry_valid then
    raise exception 'snapshot payload geometry_valid must match indexed geometry_valid';
  end if;

  if (new.payload #>> '{geometry,court_visible}')::boolean
       is distinct from new.court_visible then
    raise exception 'snapshot payload court_visible must match indexed court_visible';
  end if;

  if new.payload ->> 'source_runtime_version'
       is distinct from new.source_runtime_version then
    raise exception 'snapshot payload source_runtime_version must match indexed value';
  end if;

  eligibility_state := new.payload #>> '{spatial_eligibility,state}';
  if eligibility_state = 'invalid'
     and (
       jsonb_array_length(new.payload -> 'relationships') > 0
       or jsonb_array_length(new.payload -> 'team_shapes') > 0
       or jsonb_array_length(new.payload -> 'motion_primitives') > 0
     ) then
    raise exception 'INVALID spatial snapshot cannot persist derived metric truth';
  end if;

  return new;
end;
$$;

revoke all on function private.ensure_p4_spatial_snapshot_parent() from public, anon, authenticated;

create trigger p4_spatial_snapshots_parent_consistency
  before insert or update
  on public.p4_spatial_snapshots
  for each row execute function private.ensure_p4_spatial_snapshot_parent();


create or replace function private.ensure_p4_spatial_artifact_parent()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  parent_workspace_id uuid;
  parent_game_id uuid;
  parent_analysis_run_id uuid;
  artifact_analysis_run_id uuid;
begin
  select sr.workspace_id, sr.game_id, sr.analysis_run_id
    into parent_workspace_id, parent_game_id, parent_analysis_run_id
  from public.p4_spatial_runs sr
  where sr.id = new.spatial_run_id;

  if not found then
    raise exception 'spatial_run_id % does not exist', new.spatial_run_id;
  end if;

  if new.workspace_id is distinct from parent_workspace_id
     or new.game_id is distinct from parent_game_id
     or new.analysis_run_id is distinct from parent_analysis_run_id then
    raise exception 'p4_spatial_artifacts lineage must match p4_spatial_runs';
  end if;

  select aa.run_id
    into artifact_analysis_run_id
  from public.analysis_artifacts aa
  where aa.id = new.artifact_id;

  if not found then
    raise exception 'artifact_id % does not exist', new.artifact_id;
  end if;

  if artifact_analysis_run_id is distinct from new.analysis_run_id then
    raise exception 'analysis_artifacts.run_id must match p4 spatial analysis_run_id';
  end if;

  return new;
end;
$$;

revoke all on function private.ensure_p4_spatial_artifact_parent() from public, anon, authenticated;

create trigger p4_spatial_artifacts_parent_consistency
  before insert or update
  on public.p4_spatial_artifacts
  for each row execute function private.ensure_p4_spatial_artifact_parent();


create or replace function public.start_p4_spatial_run(
  p_id uuid,
  p_workspace_id uuid,
  p_game_id uuid,
  p_analysis_run_id uuid,
  p_contract_version text,
  p_spatial_runtime_version text,
  p_configuration_version text,
  p_fingerprint text,
  p_source_versions jsonb default '{}'::jsonb,
  p_configuration jsonb default '{}'::jsonb,
  p_provenance jsonb default '{}'::jsonb
)
returns uuid
language plpgsql
security invoker
set search_path = ''
as $$
declare
  result_id uuid;
begin
  if jsonb_typeof(coalesce(p_source_versions, '{}'::jsonb)) <> 'object'
     or jsonb_typeof(coalesce(p_configuration, '{}'::jsonb)) <> 'object'
     or jsonb_typeof(coalesce(p_provenance, '{}'::jsonb)) <> 'object' then
    raise exception 'P4 spatial source_versions/configuration/provenance must be JSON objects';
  end if;

  insert into public.p4_spatial_runs (
    id,
    workspace_id,
    game_id,
    analysis_run_id,
    contract_version,
    spatial_runtime_version,
    configuration_version,
    fingerprint,
    source_versions,
    configuration,
    provenance,
    status
  )
  values (
    p_id,
    p_workspace_id,
    p_game_id,
    p_analysis_run_id,
    p_contract_version,
    p_spatial_runtime_version,
    p_configuration_version,
    p_fingerprint,
    coalesce(p_source_versions, '{}'::jsonb),
    coalesce(p_configuration, '{}'::jsonb),
    coalesce(p_provenance, '{}'::jsonb),
    'building'
  )
  on conflict (analysis_run_id, fingerprint) do nothing;

  select sr.id
    into result_id
  from public.p4_spatial_runs sr
  where sr.analysis_run_id = p_analysis_run_id
    and sr.fingerprint = p_fingerprint;

  if result_id is null then
    raise exception 'Unable to create or resolve P4 spatial run';
  end if;

  return result_id;
end;
$$;


create or replace function public.complete_p4_spatial_run(
  p_spatial_run_id uuid,
  p_output_summary jsonb default '{}'::jsonb
)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $$
declare
  target_analysis_run_id uuid;
  previous_current_id uuid;
begin
  if jsonb_typeof(coalesce(p_output_summary, '{}'::jsonb)) <> 'object' then
    raise exception 'P4 spatial output_summary must be a JSON object';
  end if;

  select sr.analysis_run_id
    into target_analysis_run_id
  from public.p4_spatial_runs sr
  where sr.id = p_spatial_run_id
  for update;

  if not found then
    return false;
  end if;

  perform pg_advisory_xact_lock(hashtextextended(target_analysis_run_id::text, 0));

  select sr.id
    into previous_current_id
  from public.p4_spatial_runs sr
  where sr.analysis_run_id = target_analysis_run_id
    and sr.status = 'current'
    and sr.id <> p_spatial_run_id
  limit 1
  for update;

  if previous_current_id is not null then
    update public.p4_spatial_runs
    set status = 'superseded',
        completed_at = coalesce(completed_at, now())
    where id = previous_current_id;
  end if;

  update public.p4_spatial_runs
  set status = 'current',
      supersedes_id = previous_current_id,
      output_summary = coalesce(p_output_summary, '{}'::jsonb),
      error = null,
      completed_at = now()
  where id = p_spatial_run_id;

  return true;
end;
$$;


create or replace function public.fail_p4_spatial_run(
  p_spatial_run_id uuid,
  p_error text
)
returns boolean
language plpgsql
security invoker
set search_path = ''
as $$
begin
  if p_error is null or length(trim(p_error)) = 0 then
    raise exception 'P4 spatial failure reason must be non-empty';
  end if;

  update public.p4_spatial_runs
  set status = 'failed',
      error = left(p_error, 2000),
      completed_at = now()
  where id = p_spatial_run_id;

  return found;
end;
$$;


alter table public.p4_spatial_runs enable row level security;
alter table public.p4_spatial_snapshots enable row level security;
alter table public.p4_spatial_artifacts enable row level security;

create policy p4_spatial_runs_select
  on public.p4_spatial_runs
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));

create policy p4_spatial_snapshots_select
  on public.p4_spatial_snapshots
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));

create policy p4_spatial_artifacts_select
  on public.p4_spatial_artifacts
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));


revoke all on public.p4_spatial_runs from public, anon, authenticated;
revoke all on public.p4_spatial_snapshots from public, anon, authenticated;
revoke all on public.p4_spatial_artifacts from public, anon, authenticated;

grant select on public.p4_spatial_runs to authenticated;
grant select on public.p4_spatial_snapshots to authenticated;
grant select on public.p4_spatial_artifacts to authenticated;

grant select, insert, update, delete on public.p4_spatial_runs to service_role;
grant select, insert, update, delete on public.p4_spatial_snapshots to service_role;
grant select, insert, update, delete on public.p4_spatial_artifacts to service_role;

revoke all on function public.start_p4_spatial_run(
  uuid, uuid, uuid, uuid, text, text, text, text, jsonb, jsonb, jsonb
) from public, anon, authenticated;
revoke all on function public.complete_p4_spatial_run(uuid, jsonb)
  from public, anon, authenticated;
revoke all on function public.fail_p4_spatial_run(uuid, text)
  from public, anon, authenticated;

grant execute on function public.start_p4_spatial_run(
  uuid, uuid, uuid, uuid, text, text, text, text, jsonb, jsonb, jsonb
) to service_role;
grant execute on function public.complete_p4_spatial_run(uuid, jsonb)
  to service_role;
grant execute on function public.fail_p4_spatial_run(uuid, text)
  to service_role;


create trigger p4_spatial_runs_set_updated_at
  before update on public.p4_spatial_runs
  for each row execute function private.set_updated_at();


comment on table public.p4_spatial_runs is
  'Versioned P1-only P4 spatial materialization runs. No P2/P3/tactical semantics are permitted.';
comment on table public.p4_spatial_snapshots is
  'Sparse/queryable p4.spatial_snapshot.v1 materializations. Raw p4.spatial_frame.v1 remains the source contract; dense timelines may live in analysis_artifacts.';
comment on table public.p4_spatial_artifacts is
  'Lineage links from a P4 spatial run to raw-frame or materialized-snapshot analysis_artifacts belonging to the same base analysis_run.';
comment on column public.p4_spatial_runs.fingerprint is
  'Deterministic fingerprint of the base analysis run, P4 spatial contract/runtime, source versions and configuration.';
