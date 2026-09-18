-- P3 foundation: atomically select/reuse or replace the current derivation for a key.
-- The RPC is service-role only so browser clients cannot manufacture analysis lineage.

create or replace function public.start_analysis_derivation(
  p_workspace_id uuid,
  p_game_id uuid,
  p_analysis_run_id uuid,
  p_layer text,
  p_derivation_key text,
  p_contract_version text,
  p_source_contract_version text,
  p_derivation_version text,
  p_configuration_version text,
  p_fingerprint text,
  p_source_versions jsonb default '{}'::jsonb,
  p_configuration jsonb default '{}'::jsonb,
  p_provenance jsonb default '{}'::jsonb
)
returns uuid
language plpgsql
security invoker
set search_path = public, private, pg_temp
as $$
declare
  current_id uuid;
  current_fingerprint text;
  new_id uuid;
begin
  if not exists (
    select 1
    from public.analysis_runs ar
    where ar.id = p_analysis_run_id
      and ar.workspace_id = p_workspace_id
      and ar.game_id = p_game_id
  ) then
    raise exception 'analysis run does not belong to workspace/game';
  end if;

  -- Serialize first-write and replacement races for the same derivation key. Row locks
  -- alone cannot protect the first insert because there may be no current row yet.
  perform pg_advisory_xact_lock(
    hashtextextended(
      p_analysis_run_id::text || ':' || p_layer || ':' || p_derivation_key,
      0
    )
  );

  select d.id, d.fingerprint
  into current_id, current_fingerprint
  from public.analysis_derivations d
  where d.analysis_run_id = p_analysis_run_id
    and d.layer = p_layer
    and d.derivation_key = p_derivation_key
    and d.status = 'current'
  for update;

  if current_id is not null and current_fingerprint = p_fingerprint then
    return current_id;
  end if;

  if current_id is not null then
    update public.analysis_derivations
    set
      status = 'stale',
      stale_reason = 'source_version_contract_or_configuration_changed'
    where id = current_id;
  end if;

  insert into public.analysis_derivations (
    workspace_id,
    game_id,
    analysis_run_id,
    layer,
    derivation_key,
    contract_version,
    source_contract_version,
    derivation_version,
    configuration_version,
    fingerprint,
    source_versions,
    configuration,
    provenance,
    status,
    supersedes_id
  ) values (
    p_workspace_id,
    p_game_id,
    p_analysis_run_id,
    p_layer,
    p_derivation_key,
    p_contract_version,
    p_source_contract_version,
    p_derivation_version,
    p_configuration_version,
    p_fingerprint,
    coalesce(p_source_versions, '{}'::jsonb),
    coalesce(p_configuration, '{}'::jsonb),
    coalesce(p_provenance, '{}'::jsonb),
    'current',
    current_id
  )
  returning id into new_id;

  if current_id is not null then
    update public.analysis_derivations
    set status = 'superseded'
    where id = current_id;
  end if;

  return new_id;
end;
$$;

revoke all on function public.start_analysis_derivation(
  uuid, uuid, uuid, text, text, text, text, text, text, text, jsonb, jsonb, jsonb
) from public, anon, authenticated;

grant execute on function public.start_analysis_derivation(
  uuid, uuid, uuid, text, text, text, text, text, text, text, jsonb, jsonb, jsonb
) to service_role;

comment on function public.start_analysis_derivation(
  uuid, uuid, uuid, text, text, text, text, text, text, text, jsonb, jsonb, jsonb
) is 'Service-role-only atomic lifecycle for versioned/reprocessable P3+ derivations.';
