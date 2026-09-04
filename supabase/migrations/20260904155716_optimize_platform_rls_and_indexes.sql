create index analysis_runs_created_by_idx on public.analysis_runs(created_by);
create index analysis_runs_model_version_idx on public.analysis_runs(model_version_id);
create index games_home_team_idx on public.games(home_team_id);
create index games_away_team_idx on public.games(away_team_id);
create index games_created_by_idx on public.games(created_by);
create index plan_prices_plan_code_idx on public.plan_prices(plan_code);
create index subscription_events_subscription_idx on public.subscription_events(subscription_id);
create index subscription_events_workspace_idx on public.subscription_events(workspace_id);
create index subscriptions_billing_customer_idx on public.subscriptions(billing_customer_id);
create index subscriptions_plan_code_idx on public.subscriptions(plan_code);
create index subscriptions_plan_price_idx on public.subscriptions(plan_price_id);
create index usage_events_analysis_run_idx on public.usage_events(analysis_run_id);
create index usage_events_user_idx on public.usage_events(user_id);
create index workspaces_owner_user_idx on public.workspaces(owner_user_id);

alter policy profiles_select on public.profiles
  using (id = (select auth.uid()) or private.shares_workspace(id));
alter policy profiles_update on public.profiles
  using (id = (select auth.uid())) with check (id = (select auth.uid()));
alter policy profiles_insert on public.profiles
  with check (id = (select auth.uid()));

alter policy workspaces_insert on public.workspaces
  with check (owner_user_id = (select auth.uid()));
alter policy workspaces_delete on public.workspaces
  using (owner_user_id = (select auth.uid()));

alter policy workspace_members_insert on public.workspace_members
  with check (
    private.is_workspace_admin(workspace_id)
    or (
      user_id = (select auth.uid())
      and role = 'owner'
      and private.is_workspace_owner(workspace_id)
    )
  );

alter policy games_insert on public.games
  with check (
    private.can_manage_workspace(workspace_id)
    and (created_by is null or created_by = (select auth.uid()))
  );

alter policy analysis_runs_insert on public.analysis_runs
  with check (
    private.can_manage_workspace(workspace_id)
    and created_by = (select auth.uid())
  );

alter policy analysis_jobs_insert on public.analysis_jobs
  with check (
    private.can_manage_workspace(workspace_id)
    and exists (
      select 1 from public.analysis_runs r
      where r.id = run_id
        and r.workspace_id = analysis_jobs.workspace_id
        and r.created_by = (select auth.uid())
    )
  );
