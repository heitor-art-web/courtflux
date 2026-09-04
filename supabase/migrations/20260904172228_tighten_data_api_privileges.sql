-- CourtFlux least-privilege Data API grants.
-- RLS remains the row-level boundary; grants also restrict which operations
-- each API role can attempt at the table level.

revoke all privileges on all tables in schema public from anon, authenticated;
revoke all privileges on all sequences in schema public from anon, authenticated;
revoke execute on all functions in schema public from anon, authenticated, public;

-- Public pricing catalogue only.
grant select on public.plans, public.plan_prices to anon;

-- Signed-in user/profile/workspace operations.
grant select, insert, update on public.profiles to authenticated;
grant select, insert, update, delete on public.workspaces to authenticated;
grant select, insert, update, delete on public.workspace_members to authenticated;

-- Billing data is readable by members through RLS; writes are server/webhook only.
grant select on public.plans, public.plan_prices to authenticated;
grant select on public.billing_customers, public.subscriptions, public.subscription_events to authenticated;
grant select on public.workspace_entitlements to authenticated;

-- Product data managed by workspace roles through RLS.
grant select, insert, update, delete on public.teams, public.games to authenticated;
grant select on public.model_versions to authenticated;

-- Analysis runs/jobs are created by signed-in users (and currently orchestrated server-side),
-- while worker updates/results remain service-role only.
grant select, insert on public.analysis_runs, public.analysis_jobs to authenticated;
grant select on public.analysis_metrics, public.analysis_artifacts, public.usage_events to authenticated;

-- Prevent future public-schema objects from silently inheriting broad API grants.
alter default privileges for role postgres in schema public
  revoke all privileges on tables from anon, authenticated;
alter default privileges for role postgres in schema public
  revoke all privileges on sequences from anon, authenticated;
alter default privileges for role postgres in schema public
  revoke execute on functions from anon, authenticated, public;
