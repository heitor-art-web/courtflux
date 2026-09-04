create or replace function private.is_workspace_owner(target_workspace uuid)
returns boolean
language sql
stable
security definer
set search_path = public, pg_temp
as $$
  select exists (
    select 1 from public.workspaces w
    where w.id = target_workspace and w.owner_user_id = auth.uid()
  );
$$;
revoke all on function private.is_workspace_owner(uuid) from public, anon;
grant execute on function private.is_workspace_owner(uuid) to authenticated;

alter policy workspace_members_insert on public.workspace_members
  with check (
    private.is_workspace_admin(workspace_id)
    or (
      user_id = auth.uid()
      and role = 'owner'
      and private.is_workspace_owner(workspace_id)
    )
  );

create policy analysis_runs_insert on public.analysis_runs for insert to authenticated
  with check (
    private.can_manage_workspace(workspace_id)
    and created_by = auth.uid()
  );

create policy analysis_jobs_insert on public.analysis_jobs for insert to authenticated
  with check (
    private.can_manage_workspace(workspace_id)
    and exists (
      select 1 from public.analysis_runs r
      where r.id = run_id
        and r.workspace_id = analysis_jobs.workspace_id
        and r.created_by = auth.uid()
    )
  );

create or replace view public.workspace_entitlements
with (security_invoker = true)
as
select
  s.workspace_id,
  s.id as subscription_id,
  s.plan_code,
  p.name as plan_name,
  s.status,
  s.is_current,
  s.current_period_start,
  s.current_period_end,
  s.trial_end,
  s.cancel_at_period_end,
  p.features,
  p.limits,
  pp.billing_interval,
  pp.currency,
  pp.amount_cents
from public.subscriptions s
join public.plans p on p.code = s.plan_code
left join public.plan_prices pp on pp.id = s.plan_price_id
where s.is_current;

grant select on public.workspace_entitlements to authenticated;
