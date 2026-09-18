-- P3 foundation: durable lineage for configurable/reprocessable derived analysis.
--
-- P2 remains the basketball-truth source. P3 derivations record exactly which source
-- contract/runtime/configuration produced a derived dataset so later P2 calibration can
-- mark old output stale and rebuild it without changing the Game identity or source run.

create table if not exists public.analysis_derivations (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid references public.games(id) on delete cascade,
  analysis_run_id uuid not null references public.analysis_runs(id) on delete cascade,

  layer text not null default 'p3' check (layer in ('p3','p4','p5')),
  derivation_key text not null,

  contract_version text not null,
  source_contract_version text not null,
  derivation_version text not null,
  configuration_version text not null,
  fingerprint text not null,

  source_versions jsonb not null default '{}'::jsonb,
  configuration jsonb not null default '{}'::jsonb,
  provenance jsonb not null default '{}'::jsonb,
  output_summary jsonb not null default '{}'::jsonb,

  status text not null default 'current'
    check (status in ('current','stale','superseded','failed')),
  supersedes_id uuid references public.analysis_derivations(id) on delete set null,
  stale_reason text,

  generated_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  check (length(trim(derivation_key)) > 0),
  check (length(trim(contract_version)) > 0),
  check (length(trim(source_contract_version)) > 0),
  check (length(trim(derivation_version)) > 0),
  check (length(trim(configuration_version)) > 0),
  check (length(trim(fingerprint)) > 0)
);

create index if not exists analysis_derivations_run_idx
  on public.analysis_derivations(analysis_run_id, layer, derivation_key, created_at desc);

create index if not exists analysis_derivations_game_idx
  on public.analysis_derivations(game_id, layer, derivation_key, created_at desc)
  where game_id is not null;

create index if not exists analysis_derivations_status_idx
  on public.analysis_derivations(workspace_id, status, created_at desc);

-- Only one derivation is authoritative for a run/layer/key at a time. Historical rows
-- remain addressable as stale/superseded evidence instead of being overwritten.
create unique index if not exists analysis_derivations_one_current_idx
  on public.analysis_derivations(analysis_run_id, layer, derivation_key)
  where status = 'current';

alter table public.analysis_derivations enable row level security;

create policy analysis_derivations_select
  on public.analysis_derivations
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));

-- Do not depend on project default privileges for exposed-schema objects.
revoke all on public.analysis_derivations from public, anon, authenticated;
grant select on public.analysis_derivations to authenticated;
grant select, insert, update, delete on public.analysis_derivations to service_role;

create trigger analysis_derivations_set_updated_at
  before update on public.analysis_derivations
  for each row execute function private.set_updated_at();

comment on table public.analysis_derivations is
  'Versioned lineage records for reprocessable P3+ derived analysis. Historical derivations are retained rather than silently overwritten.';
comment on column public.analysis_derivations.fingerprint is
  'Deterministic fingerprint of source versions, contracts and configuration used for this derivation.';
comment on column public.analysis_derivations.status is
  'current is authoritative; stale/superseded remain historical and explainable; failed records unsuccessful derivation attempts.';
