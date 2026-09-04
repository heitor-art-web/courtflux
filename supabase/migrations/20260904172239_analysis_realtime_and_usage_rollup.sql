-- Make analysis state observable from the app without polling the worker directly.
do $$
begin
  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public'
      and tablename = 'analysis_runs'
  ) then
    alter publication supabase_realtime add table public.analysis_runs;
  end if;

  if not exists (
    select 1 from pg_publication_tables
    where pubname = 'supabase_realtime'
      and schemaname = 'public'
      and tablename = 'analysis_jobs'
  ) then
    alter publication supabase_realtime add table public.analysis_jobs;
  end if;
end
$$;

-- Subscription metering/read model. security_invoker keeps usage_events RLS
-- active for the signed-in caller.
create or replace view public.workspace_usage_monthly
with (security_invoker = true)
as
select
  workspace_id,
  date_trunc('month', occurred_at) as usage_month,
  metric_code,
  sum(quantity)::numeric as quantity
from public.usage_events
group by workspace_id, date_trunc('month', occurred_at), metric_code;

grant select on public.workspace_usage_monthly to authenticated;
