-- P3.3 hardening: durable source identity, clip request lineage, and retry-safe
-- generation lifecycle. These changes are independent from P2 basketball accuracy.

alter table public.p3_clips
  add column if not exists source_video_ref text,
  add column if not exists source_possession_id text,
  add column if not exists request_contract_version text,
  add column if not exists request_fingerprint text,
  add column if not exists generation_attempts integer not null default 0,
  add column if not exists last_generation_started_at timestamptz,
  add column if not exists last_generation_finished_at timestamptz,
  add column if not exists ready_at timestamptz;

update public.p3_clips
set source_video_ref = 'analysis-run:' || analysis_run_id::text
where source_video_ref is null or length(trim(source_video_ref)) = 0;

update public.p3_clips
set request_contract_version = 'p3.clip_request.v1'
where request_contract_version is null or length(trim(request_contract_version)) = 0;

update public.p3_clips
set ready_at = coalesce(ready_at, updated_at, created_at)
where status = 'ready' and artifact_id is not null;

update public.p3_clips
set status = 'failed',
    error = coalesce(error, 'legacy ready clip had no durable artifact linkage'),
    ready_at = null
where status = 'ready' and artifact_id is null;

alter table public.p3_clips
  alter column source_video_ref set not null,
  alter column request_contract_version set not null;

alter table public.p3_clips
  add constraint p3_clips_source_video_ref_nonempty
    check (length(trim(source_video_ref)) > 0),
  add constraint p3_clips_request_contract_nonempty
    check (length(trim(request_contract_version)) > 0),
  add constraint p3_clips_request_fingerprint_shape
    check (
      request_fingerprint is null
      or request_fingerprint ~ '^[0-9a-f]{64}$'
    ),
  add constraint p3_clips_positive_duration
    check (clip_end_ms > clip_start_ms),
  add constraint p3_clips_contains_source_interval
    check (
      clip_start_ms <= source_start_ms
      and clip_end_ms >= source_end_ms
    ),
  add constraint p3_clips_generation_attempts_nonnegative
    check (generation_attempts >= 0),
  add constraint p3_clips_generation_time_order
    check (
      last_generation_finished_at is null
      or last_generation_started_at is null
      or last_generation_finished_at >= last_generation_started_at
    ),
  add constraint p3_clips_ready_artifact_consistency
    check (
      (
        status = 'ready'
        and artifact_id is not null
        and error is null
        and ready_at is not null
      )
      or
      (
        status <> 'ready'
        and artifact_id is null
        and ready_at is null
      )
    );

create index if not exists p3_clips_source_ref_idx
  on public.p3_clips(analysis_run_id, source_video_ref);

create or replace function public.begin_p3_clip_generation(
  p_clip_id uuid,
  p_lease_seconds integer default 900
)
returns boolean
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_id uuid;
  v_lease_seconds integer := greatest(60, coalesce(p_lease_seconds, 900));
begin
  update public.p3_clips
  set status = 'generating',
      artifact_id = null,
      error = null,
      ready_at = null,
      generation_attempts = generation_attempts + 1,
      last_generation_started_at = now(),
      last_generation_finished_at = null
  where id = p_clip_id
    and (
      status in ('planned', 'failed')
      or (
        status = 'generating'
        and (
          last_generation_started_at is null
          or last_generation_started_at < now() - make_interval(secs => v_lease_seconds)
        )
      )
      or (status = 'ready' and artifact_id is null)
    )
  returning id into v_id;

  return v_id is not null;
end;
$$;

create or replace function public.complete_p3_clip_generation(
  p_clip_id uuid,
  p_artifact_id uuid
)
returns boolean
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_run_id uuid;
  v_status text;
  v_existing_artifact uuid;
  v_request_fingerprint text;
begin
  select analysis_run_id, status, artifact_id, request_fingerprint
    into v_run_id, v_status, v_existing_artifact, v_request_fingerprint
  from public.p3_clips
  where id = p_clip_id;

  if not found then
    raise exception 'P3 clip % does not exist', p_clip_id;
  end if;

  if v_status = 'ready' and v_existing_artifact = p_artifact_id then
    return true;
  end if;

  if v_status <> 'generating' then
    return false;
  end if;

  if not exists (
    select 1
    from public.analysis_artifacts a
    where a.id = p_artifact_id
      and a.run_id = v_run_id
      and a.kind = 'p3_evidence_clip'
      and a.bucket = 'analysis-artifacts'
      and a.metadata->>'clip_id' = p_clip_id::text
      and (
        v_request_fingerprint is null
        or a.metadata->>'request_fingerprint' = v_request_fingerprint
      )
  ) then
    raise exception 'Artifact % is not a valid durable artifact for P3 clip %',
      p_artifact_id, p_clip_id;
  end if;

  update public.p3_clips
  set status = 'ready',
      artifact_id = p_artifact_id,
      error = null,
      last_generation_finished_at = now(),
      ready_at = now()
  where id = p_clip_id
    and status = 'generating';

  return found;
end;
$$;

create or replace function public.fail_p3_clip_generation(
  p_clip_id uuid,
  p_error text
)
returns boolean
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
begin
  update public.p3_clips
  set status = 'failed',
      artifact_id = null,
      error = left(coalesce(nullif(trim(p_error), ''), 'clip generation failed'), 2000),
      ready_at = null,
      last_generation_finished_at = now()
  where id = p_clip_id
    and status = 'generating';

  return found;
end;
$$;

revoke all on function public.begin_p3_clip_generation(uuid, integer)
  from public, anon, authenticated;
revoke all on function public.complete_p3_clip_generation(uuid, uuid)
  from public, anon, authenticated;
revoke all on function public.fail_p3_clip_generation(uuid, text)
  from public, anon, authenticated;

grant execute on function public.begin_p3_clip_generation(uuid, integer)
  to service_role;
grant execute on function public.complete_p3_clip_generation(uuid, uuid)
  to service_role;
grant execute on function public.fail_p3_clip_generation(uuid, text)
  to service_role;

comment on column public.p3_clips.source_video_ref is
  'Stable logical source-video identity. Default analysis-run:<run_id> resolves through analysis_runs.video_source.';
comment on column public.p3_clips.request_fingerprint is
  'SHA-256 of the canonical p3.clip_request.v1 payload for deterministic regeneration identity.';
comment on column public.p3_clips.generation_attempts is
  'Number of generation leases acquired for this deterministic clip request.';
comment on function public.begin_p3_clip_generation(uuid, integer) is
  'Acquire or recover a P3 clip-generation lease. Service-role only.';
comment on function public.complete_p3_clip_generation(uuid, uuid) is
  'Atomically mark a generating P3 clip ready only when its durable artifact lineage is valid.';
comment on function public.fail_p3_clip_generation(uuid, text) is
  'Mark the current P3 clip-generation attempt failed without deleting the review item or source interval.';
