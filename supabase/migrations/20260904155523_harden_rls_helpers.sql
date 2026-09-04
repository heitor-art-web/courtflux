create schema if not exists private;
revoke all on schema private from public;
grant usage on schema private to authenticated;

create or replace function private.is_workspace_member(target_workspace uuid)
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

create or replace function private.workspace_role(target_workspace uuid)
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

create or replace function private.can_manage_workspace(target_workspace uuid)
returns boolean
language sql
stable
security definer
set search_path = public, private, pg_temp
as $$
  select coalesce(private.workspace_role(target_workspace) in ('owner','admin','analyst'), false);
$$;

create or replace function private.is_workspace_admin(target_workspace uuid)
returns boolean
language sql
stable
security definer
set search_path = public, private, pg_temp
as $$
  select coalesce(private.workspace_role(target_workspace) in ('owner','admin'), false);
$$;

create or replace function private.shares_workspace(target_user uuid)
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

grant execute on function private.is_workspace_member(uuid) to authenticated;
grant execute on function private.workspace_role(uuid) to authenticated;
grant execute on function private.can_manage_workspace(uuid) to authenticated;
grant execute on function private.is_workspace_admin(uuid) to authenticated;
grant execute on function private.shares_workspace(uuid) to authenticated;

alter policy profiles_select on public.profiles
  using (id = auth.uid() or private.shares_workspace(id));

alter policy workspaces_select on public.workspaces
  using (private.is_workspace_member(id));
alter policy workspaces_update on public.workspaces
  using (private.is_workspace_admin(id)) with check (private.is_workspace_admin(id));

alter policy workspace_members_select on public.workspace_members
  using (private.is_workspace_member(workspace_id));
alter policy workspace_members_insert on public.workspace_members
  with check (
    private.is_workspace_admin(workspace_id)
    or (
      user_id = auth.uid()
      and role = 'owner'
      and exists (
        select 1 from public.workspaces w
        where w.id = workspace_id and w.owner_user_id = auth.uid()
      )
    )
  );
alter policy workspace_members_update on public.workspace_members
  using (private.is_workspace_admin(workspace_id)) with check (private.is_workspace_admin(workspace_id));
alter policy workspace_members_delete on public.workspace_members
  using (private.is_workspace_admin(workspace_id));

alter policy billing_customers_select on public.billing_customers
  using (private.is_workspace_member(workspace_id));
alter policy subscriptions_select on public.subscriptions
  using (private.is_workspace_member(workspace_id));
alter policy subscription_events_select on public.subscription_events
  using (private.is_workspace_member(workspace_id));

alter policy teams_select on public.teams
  using (private.is_workspace_member(workspace_id));
alter policy teams_insert on public.teams
  with check (private.can_manage_workspace(workspace_id));
alter policy teams_update on public.teams
  using (private.can_manage_workspace(workspace_id)) with check (private.can_manage_workspace(workspace_id));
alter policy teams_delete on public.teams
  using (private.is_workspace_admin(workspace_id));

alter policy games_select on public.games
  using (private.is_workspace_member(workspace_id));
alter policy games_insert on public.games
  with check (private.can_manage_workspace(workspace_id) and (created_by is null or created_by = auth.uid()));
alter policy games_update on public.games
  using (private.can_manage_workspace(workspace_id)) with check (private.can_manage_workspace(workspace_id));
alter policy games_delete on public.games
  using (private.is_workspace_admin(workspace_id));

alter policy analysis_runs_select on public.analysis_runs
  using (private.is_workspace_member(workspace_id));
alter policy analysis_jobs_select on public.analysis_jobs
  using (private.is_workspace_member(workspace_id));
alter policy analysis_metrics_select on public.analysis_metrics
  using (exists (
    select 1 from public.analysis_runs r
    where r.id = run_id and private.is_workspace_member(r.workspace_id)
  ));
alter policy analysis_artifacts_select on public.analysis_artifacts
  using (exists (
    select 1 from public.analysis_runs r
    where r.id = run_id and private.is_workspace_member(r.workspace_id)
  ));
alter policy usage_events_select on public.usage_events
  using (private.is_workspace_member(workspace_id));

create or replace function private.set_updated_at()
returns trigger
language plpgsql
set search_path = public, pg_temp
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

alter function public.handle_new_user() set search_path = public, pg_temp;
alter function public.handle_user_updated() set search_path = public, pg_temp;
alter function public.set_updated_at() set search_path = public, pg_temp;

revoke all on function public.handle_new_user() from public, anon, authenticated;
revoke all on function public.handle_user_updated() from public, anon, authenticated;
revoke all on function public.set_updated_at() from public, anon, authenticated;
revoke all on function public.is_workspace_member(uuid) from public, anon, authenticated;
revoke all on function public.workspace_role(uuid) from public, anon, authenticated;
revoke all on function public.can_manage_workspace(uuid) from public, anon, authenticated;
revoke all on function public.is_workspace_admin(uuid) from public, anon, authenticated;
revoke all on function public.shares_workspace(uuid) from public, anon, authenticated;
