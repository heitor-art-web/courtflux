create extension if not exists pgcrypto with schema extensions;

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  avatar_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.workspaces (
  id uuid primary key default gen_random_uuid(),
  owner_user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  slug text not null unique,
  kind text not null default 'personal' check (kind in ('personal','team')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.workspace_members (
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null default 'analyst' check (role in ('owner','admin','analyst','viewer')),
  status text not null default 'active' check (status in ('active','inactive')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (workspace_id, user_id)
);

create table public.plans (
  code text primary key,
  name text not null,
  description text,
  is_active boolean not null default true,
  sort_order integer not null default 0,
  features jsonb not null default '{}'::jsonb,
  limits jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.plan_prices (
  id uuid primary key default gen_random_uuid(),
  plan_code text not null references public.plans(code) on delete cascade,
  billing_interval text not null check (billing_interval in ('month','year')),
  currency text not null default 'EUR',
  amount_cents integer check (amount_cents is null or amount_cents >= 0),
  provider text,
  provider_price_id text,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (provider, provider_price_id)
);

create table public.billing_customers (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  provider text not null,
  provider_customer_id text not null,
  email text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (workspace_id, provider),
  unique (provider, provider_customer_id)
);

create table public.subscriptions (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  plan_code text not null references public.plans(code),
  plan_price_id uuid references public.plan_prices(id),
  billing_customer_id uuid references public.billing_customers(id) on delete set null,
  provider text,
  provider_subscription_id text,
  status text not null default 'active' check (status in ('free','trialing','active','past_due','paused','canceled','incomplete','incomplete_expired')),
  is_current boolean not null default true,
  current_period_start timestamptz,
  current_period_end timestamptz,
  trial_end timestamptz,
  cancel_at_period_end boolean not null default false,
  canceled_at timestamptz,
  ended_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (provider, provider_subscription_id)
);
create unique index subscriptions_one_current_per_workspace
  on public.subscriptions(workspace_id) where is_current;

create table public.subscription_events (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  subscription_id uuid references public.subscriptions(id) on delete set null,
  provider text,
  provider_event_id text,
  event_type text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (provider, provider_event_id)
);

create table public.teams (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  name text not null,
  short_name text,
  external_id text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (workspace_id, name)
);

create table public.games (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  created_by uuid references auth.users(id) on delete set null,
  title text,
  competition text,
  season text,
  game_date timestamptz,
  home_team_id uuid references public.teams(id) on delete set null,
  away_team_id uuid references public.teams(id) on delete set null,
  video_source text,
  source_type text not null default 'video' check (source_type in ('video','youtube','stream','upload','other')),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (home_team_id is null or away_team_id is null or home_team_id <> away_team_id)
);

create table public.model_versions (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  version text not null,
  artifact_path text,
  checksum text,
  status text not null default 'active' check (status in ('draft','active','deprecated')),
  config jsonb not null default '{}'::jsonb,
  evaluation_metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (name, version)
);

create table public.analysis_runs (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid references public.games(id) on delete set null,
  created_by uuid references auth.users(id) on delete set null,
  model_version_id uuid references public.model_versions(id) on delete set null,
  status text not null default 'queued' check (status in ('queued','running','completed','failed','canceled')),
  video_source text,
  start_time_sec double precision not null default 0 check (start_time_sec >= 0),
  end_time_sec double precision check (end_time_sec is null or end_time_sec > start_time_sec),
  parameters jsonb not null default '{}'::jsonb,
  summary jsonb not null default '{}'::jsonb,
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.analysis_jobs (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null unique references public.analysis_runs(id) on delete cascade,
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  status text not null default 'queued' check (status in ('queued','running','completed','failed','canceled')),
  worker_id text,
  request jsonb not null default '{}'::jsonb,
  error text,
  log_tail text,
  queued_at timestamptz not null default now(),
  started_at timestamptz,
  completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.analysis_metrics (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.analysis_runs(id) on delete cascade,
  scope text not null default 'run',
  metric_key text not null,
  metric_value double precision,
  metric_text text,
  unit text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (run_id, scope, metric_key),
  check (metric_value is not null or metric_text is not null)
);

create table public.analysis_artifacts (
  id uuid primary key default gen_random_uuid(),
  run_id uuid not null references public.analysis_runs(id) on delete cascade,
  kind text not null,
  bucket text,
  object_path text not null,
  content_type text,
  size_bytes bigint check (size_bytes is null or size_bytes >= 0),
  checksum text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (run_id, kind, object_path)
);

create table public.usage_events (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  user_id uuid references auth.users(id) on delete set null,
  analysis_run_id uuid references public.analysis_runs(id) on delete set null,
  metric_code text not null,
  quantity numeric not null default 1 check (quantity >= 0),
  metadata jsonb not null default '{}'::jsonb,
  occurred_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

create index workspace_members_user_idx on public.workspace_members(user_id) where status = 'active';
create index teams_workspace_idx on public.teams(workspace_id);
create index games_workspace_date_idx on public.games(workspace_id, game_date desc);
create index analysis_runs_workspace_created_idx on public.analysis_runs(workspace_id, created_at desc);
create index analysis_runs_game_idx on public.analysis_runs(game_id);
create index analysis_jobs_workspace_status_idx on public.analysis_jobs(workspace_id, status);
create index analysis_metrics_key_idx on public.analysis_metrics(metric_key, metric_value);
create index usage_events_workspace_time_idx on public.usage_events(workspace_id, occurred_at desc);

insert into public.plans (code, name, description, sort_order, features, limits)
values
  ('free', 'Free', 'Free CourtFlux access.', 10, '{"analysis":true,"history":true,"exports":false,"advanced_analytics":false,"team_workspace":false,"priority_processing":false}'::jsonb, '{}'::jsonb),
  ('base', 'Base', 'Core CourtFlux analysis for regular use.', 20, '{"analysis":true,"history":true,"exports":true,"advanced_analytics":false,"team_workspace":false,"priority_processing":false}'::jsonb, '{}'::jsonb),
  ('premium', 'Premium', 'Full CourtFlux intelligence and collaboration.', 30, '{"analysis":true,"history":true,"exports":true,"advanced_analytics":true,"team_workspace":true,"priority_processing":true}'::jsonb, '{}'::jsonb)
on conflict (code) do update set
  name = excluded.name,
  description = excluded.description,
  sort_order = excluded.sort_order,
  features = excluded.features,
  limits = excluded.limits,
  updated_at = now();

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger profiles_set_updated_at before update on public.profiles for each row execute function public.set_updated_at();
create trigger workspaces_set_updated_at before update on public.workspaces for each row execute function public.set_updated_at();
create trigger workspace_members_set_updated_at before update on public.workspace_members for each row execute function public.set_updated_at();
create trigger plans_set_updated_at before update on public.plans for each row execute function public.set_updated_at();
create trigger plan_prices_set_updated_at before update on public.plan_prices for each row execute function public.set_updated_at();
create trigger billing_customers_set_updated_at before update on public.billing_customers for each row execute function public.set_updated_at();
create trigger subscriptions_set_updated_at before update on public.subscriptions for each row execute function public.set_updated_at();
create trigger teams_set_updated_at before update on public.teams for each row execute function public.set_updated_at();
create trigger games_set_updated_at before update on public.games for each row execute function public.set_updated_at();
create trigger model_versions_set_updated_at before update on public.model_versions for each row execute function public.set_updated_at();
create trigger analysis_runs_set_updated_at before update on public.analysis_runs for each row execute function public.set_updated_at();
create trigger analysis_jobs_set_updated_at before update on public.analysis_jobs for each row execute function public.set_updated_at();

create or replace function public.is_workspace_member(target_workspace uuid)
returns boolean
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select exists (
    select 1
    from public.workspace_members wm
    where wm.workspace_id = target_workspace
      and wm.user_id = auth.uid()
      and wm.status = 'active'
  );
$$;

create or replace function public.workspace_role(target_workspace uuid)
returns text
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select wm.role
  from public.workspace_members wm
  where wm.workspace_id = target_workspace
    and wm.user_id = auth.uid()
    and wm.status = 'active'
  limit 1;
$$;

create or replace function public.can_manage_workspace(target_workspace uuid)
returns boolean
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select coalesce(public.workspace_role(target_workspace) in ('owner','admin','analyst'), false);
$$;

create or replace function public.is_workspace_admin(target_workspace uuid)
returns boolean
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select coalesce(public.workspace_role(target_workspace) in ('owner','admin'), false);
$$;

create or replace function public.shares_workspace(target_user uuid)
returns boolean
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select exists (
    select 1
    from public.workspace_members mine
    join public.workspace_members theirs on theirs.workspace_id = mine.workspace_id
    where mine.user_id = auth.uid()
      and mine.status = 'active'
      and theirs.user_id = target_user
      and theirs.status = 'active'
  );
$$;

revoke all on function public.is_workspace_member(uuid) from public;
revoke all on function public.workspace_role(uuid) from public;
revoke all on function public.can_manage_workspace(uuid) from public;
revoke all on function public.is_workspace_admin(uuid) from public;
revoke all on function public.shares_workspace(uuid) from public;
grant execute on function public.is_workspace_member(uuid) to authenticated;
grant execute on function public.workspace_role(uuid) to authenticated;
grant execute on function public.can_manage_workspace(uuid) to authenticated;
grant execute on function public.is_workspace_admin(uuid) to authenticated;
grant execute on function public.shares_workspace(uuid) to authenticated;

alter table public.profiles enable row level security;
alter table public.workspaces enable row level security;
alter table public.workspace_members enable row level security;
alter table public.plans enable row level security;
alter table public.plan_prices enable row level security;
alter table public.billing_customers enable row level security;
alter table public.subscriptions enable row level security;
alter table public.subscription_events enable row level security;
alter table public.teams enable row level security;
alter table public.games enable row level security;
alter table public.model_versions enable row level security;
alter table public.analysis_runs enable row level security;
alter table public.analysis_jobs enable row level security;
alter table public.analysis_metrics enable row level security;
alter table public.analysis_artifacts enable row level security;
alter table public.usage_events enable row level security;

create policy profiles_select on public.profiles for select to authenticated
  using (id = auth.uid() or public.shares_workspace(id));
create policy profiles_update on public.profiles for update to authenticated
  using (id = auth.uid()) with check (id = auth.uid());
create policy profiles_insert on public.profiles for insert to authenticated
  with check (id = auth.uid());

create policy workspaces_select on public.workspaces for select to authenticated
  using (public.is_workspace_member(id));
create policy workspaces_insert on public.workspaces for insert to authenticated
  with check (owner_user_id = auth.uid());
create policy workspaces_update on public.workspaces for update to authenticated
  using (public.is_workspace_admin(id)) with check (public.is_workspace_admin(id));
create policy workspaces_delete on public.workspaces for delete to authenticated
  using (owner_user_id = auth.uid());

create policy workspace_members_select on public.workspace_members for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy workspace_members_insert on public.workspace_members for insert to authenticated
  with check (
    public.is_workspace_admin(workspace_id)
    or (
      user_id = auth.uid()
      and role = 'owner'
      and exists (
        select 1 from public.workspaces w
        where w.id = workspace_id and w.owner_user_id = auth.uid()
      )
    )
  );
create policy workspace_members_update on public.workspace_members for update to authenticated
  using (public.is_workspace_admin(workspace_id)) with check (public.is_workspace_admin(workspace_id));
create policy workspace_members_delete on public.workspace_members for delete to authenticated
  using (public.is_workspace_admin(workspace_id));

create policy plans_select on public.plans for select to anon, authenticated using (is_active);
create policy plan_prices_select on public.plan_prices for select to anon, authenticated using (is_active);

create policy billing_customers_select on public.billing_customers for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy subscriptions_select on public.subscriptions for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy subscription_events_select on public.subscription_events for select to authenticated
  using (public.is_workspace_member(workspace_id));

create policy teams_select on public.teams for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy teams_insert on public.teams for insert to authenticated
  with check (public.can_manage_workspace(workspace_id));
create policy teams_update on public.teams for update to authenticated
  using (public.can_manage_workspace(workspace_id)) with check (public.can_manage_workspace(workspace_id));
create policy teams_delete on public.teams for delete to authenticated
  using (public.is_workspace_admin(workspace_id));

create policy games_select on public.games for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy games_insert on public.games for insert to authenticated
  with check (public.can_manage_workspace(workspace_id) and (created_by is null or created_by = auth.uid()));
create policy games_update on public.games for update to authenticated
  using (public.can_manage_workspace(workspace_id)) with check (public.can_manage_workspace(workspace_id));
create policy games_delete on public.games for delete to authenticated
  using (public.is_workspace_admin(workspace_id));

create policy model_versions_select on public.model_versions for select to authenticated using (true);

create policy analysis_runs_select on public.analysis_runs for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy analysis_jobs_select on public.analysis_jobs for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy analysis_metrics_select on public.analysis_metrics for select to authenticated
  using (exists (
    select 1 from public.analysis_runs r
    where r.id = run_id and public.is_workspace_member(r.workspace_id)
  ));
create policy analysis_artifacts_select on public.analysis_artifacts for select to authenticated
  using (exists (
    select 1 from public.analysis_runs r
    where r.id = run_id and public.is_workspace_member(r.workspace_id)
  ));
create policy usage_events_select on public.usage_events for select to authenticated
  using (public.is_workspace_member(workspace_id));

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
  values (new_workspace_id, 'free', 'free', true);

  return new;
end;
$$;

create or replace function public.handle_user_updated()
returns trigger
language plpgsql
security definer
set search_path = public, pg_temp
as $$
begin
  update public.profiles
  set
    email = new.email,
    full_name = coalesce(nullif(new.raw_user_meta_data ->> 'full_name', ''), full_name),
    avatar_url = coalesce(nullif(new.raw_user_meta_data ->> 'avatar_url', ''), avatar_url),
    updated_at = now()
  where id = new.id;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

drop trigger if exists on_auth_user_updated on auth.users;
create trigger on_auth_user_updated
  after update of email, raw_user_meta_data on auth.users
  for each row execute function public.handle_user_updated();
