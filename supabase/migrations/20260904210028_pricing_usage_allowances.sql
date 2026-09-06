-- Align platform entitlements with the public CourtFlux private-beta plans and
-- add a controlled way to grant temporary analysis-minute top-ups.

insert into public.plans (code, name, description, is_active, sort_order, features, limits)
values
  (
    'beta',
    'Beta',
    'Private-beta CourtFlux access for trying basketball video analysis.',
    true,
    10,
    '{"analysis":true,"history":true,"exports":false,"advanced_analytics":false,"team_workspace":false,"priority_processing":false}'::jsonb,
    '{"analysis_minutes_monthly":60,"usage_warning_percent":80,"usage_hard_stop_percent":100,"overage_billing":false}'::jsonb
  ),
  (
    'coach',
    'Coach',
    'CourtFlux analysis for an individual coach workspace.',
    true,
    20,
    '{"analysis":true,"history":true,"exports":false,"advanced_analytics":false,"team_workspace":false,"priority_processing":false}'::jsonb,
    '{"analysis_minutes_monthly":300,"usage_warning_percent":80,"usage_hard_stop_percent":100,"overage_billing":false}'::jsonb
  ),
  (
    'team',
    'Team',
    'Shared CourtFlux analysis and structured team workflows.',
    true,
    30,
    '{"analysis":true,"history":true,"exports":true,"advanced_analytics":false,"team_workspace":true,"priority_processing":false}'::jsonb,
    '{"analysis_minutes_monthly":720,"usage_warning_percent":80,"usage_hard_stop_percent":100,"overage_billing":false}'::jsonb
  ),
  (
    'academy',
    'Academy',
    'Multi-team CourtFlux analysis for academy development workflows.',
    true,
    40,
    '{"analysis":true,"history":true,"exports":true,"advanced_analytics":false,"team_workspace":true,"priority_processing":false}'::jsonb,
    '{"analysis_minutes_monthly":2400,"usage_warning_percent":80,"usage_hard_stop_percent":100,"overage_billing":false,"included_teams":5}'::jsonb
  )
on conflict (code) do update set
  name = excluded.name,
  description = excluded.description,
  is_active = excluded.is_active,
  sort_order = excluded.sort_order,
  features = excluded.features,
  limits = excluded.limits,
  updated_at = now();

-- The original Free/Base/Premium rows were scaffolding. Preserve historical rows
-- but move existing current subscriptions onto the public product names.
update public.subscriptions
set plan_code = case plan_code
  when 'free' then 'beta'
  when 'base' then 'coach'
  when 'premium' then 'team'
  else plan_code
end
where plan_code in ('free', 'base', 'premium');

update public.plans
set is_active = false, updated_at = now()
where code in ('free', 'base', 'premium');

-- New users start on the Beta plan. Billing remains inactive until a provider is
-- explicitly connected later.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  new_workspace_id uuid;
  workspace_name text;
begin
  insert into public.profiles (id, email, full_name, avatar_url)
  values (
    new.id,
    new.email,
    nullif(new.raw_user_meta_data ->> 'full_name', ''),
    nullif(new.raw_user_meta_data ->> 'avatar_url', '')
  )
  on conflict (id) do update set
    email = excluded.email,
    full_name = coalesce(excluded.full_name, public.profiles.full_name),
    avatar_url = coalesce(excluded.avatar_url, public.profiles.avatar_url),
    updated_at = now();

  workspace_name := coalesce(
    nullif(new.raw_user_meta_data ->> 'full_name', ''),
    nullif(split_part(coalesce(new.email, ''), '@', 1), ''),
    'CourtFlux'
  );

  insert into public.workspaces (owner_user_id, name, slug, kind)
  values (new.id, workspace_name || ' Workspace', 'personal-' || replace(new.id::text, '-', ''), 'personal')
  returning id into new_workspace_id;

  insert into public.workspace_members (workspace_id, user_id, role, status)
  values (new_workspace_id, new.id, 'owner', 'active');

  insert into public.subscriptions (workspace_id, plan_code, status, is_current)
  values (new_workspace_id, 'beta', 'free', true);

  return new;
end;
$$;

-- Top-ups are deliberately separate from plan limits. Authenticated workspace
-- members may read them, but only trusted server/service-role operations can
-- create, update or delete grants.
create table if not exists public.usage_allowance_grants (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  metric_code text not null default 'analysis_minutes',
  quantity numeric not null check (quantity > 0),
  valid_from timestamptz not null default now(),
  valid_until timestamptz,
  note text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  check (valid_until is null or valid_until > valid_from)
);

create index if not exists usage_allowance_grants_workspace_time_idx
  on public.usage_allowance_grants(workspace_id, metric_code, valid_from, valid_until);

alter table public.usage_allowance_grants enable row level security;

drop policy if exists usage_allowance_grants_select on public.usage_allowance_grants;
create policy usage_allowance_grants_select on public.usage_allowance_grants
  for select to authenticated
  using (private.is_workspace_member(workspace_id));

grant select on public.usage_allowance_grants to authenticated;
grant all on public.usage_allowance_grants to service_role;
