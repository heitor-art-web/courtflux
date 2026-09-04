create or replace function private.handle_new_user()
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

create or replace function private.handle_user_updated()
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

revoke all on function private.handle_new_user() from public, anon, authenticated;
revoke all on function private.handle_user_updated() from public, anon, authenticated;
revoke all on function private.set_updated_at() from public, anon, authenticated;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function private.handle_new_user();

drop trigger if exists on_auth_user_updated on auth.users;
create trigger on_auth_user_updated
  after update of email, raw_user_meta_data on auth.users
  for each row execute function private.handle_user_updated();

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at before update on public.profiles for each row execute function private.set_updated_at();
drop trigger if exists workspaces_set_updated_at on public.workspaces;
create trigger workspaces_set_updated_at before update on public.workspaces for each row execute function private.set_updated_at();
drop trigger if exists workspace_members_set_updated_at on public.workspace_members;
create trigger workspace_members_set_updated_at before update on public.workspace_members for each row execute function private.set_updated_at();
drop trigger if exists plans_set_updated_at on public.plans;
create trigger plans_set_updated_at before update on public.plans for each row execute function private.set_updated_at();
drop trigger if exists plan_prices_set_updated_at on public.plan_prices;
create trigger plan_prices_set_updated_at before update on public.plan_prices for each row execute function private.set_updated_at();
drop trigger if exists billing_customers_set_updated_at on public.billing_customers;
create trigger billing_customers_set_updated_at before update on public.billing_customers for each row execute function private.set_updated_at();
drop trigger if exists subscriptions_set_updated_at on public.subscriptions;
create trigger subscriptions_set_updated_at before update on public.subscriptions for each row execute function private.set_updated_at();
drop trigger if exists teams_set_updated_at on public.teams;
create trigger teams_set_updated_at before update on public.teams for each row execute function private.set_updated_at();
drop trigger if exists games_set_updated_at on public.games;
create trigger games_set_updated_at before update on public.games for each row execute function private.set_updated_at();
drop trigger if exists model_versions_set_updated_at on public.model_versions;
create trigger model_versions_set_updated_at before update on public.model_versions for each row execute function private.set_updated_at();
drop trigger if exists analysis_runs_set_updated_at on public.analysis_runs;
create trigger analysis_runs_set_updated_at before update on public.analysis_runs for each row execute function private.set_updated_at();
drop trigger if exists analysis_jobs_set_updated_at on public.analysis_jobs;
create trigger analysis_jobs_set_updated_at before update on public.analysis_jobs for each row execute function private.set_updated_at();

drop function if exists public.handle_new_user();
drop function if exists public.handle_user_updated();
drop function if exists public.set_updated_at();
drop function if exists public.is_workspace_member(uuid);
drop function if exists public.workspace_role(uuid);
drop function if exists public.can_manage_workspace(uuid);
drop function if exists public.is_workspace_admin(uuid);
drop function if exists public.shares_workspace(uuid);
