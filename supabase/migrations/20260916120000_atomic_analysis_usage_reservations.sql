begin;

create table if not exists public.analysis_usage_reservations (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  run_id uuid not null unique references public.analysis_runs(id) on delete cascade,
  user_id uuid references auth.users(id) on delete set null,
  metric_code text not null default 'analysis_minutes' check (metric_code = 'analysis_minutes'),
  quantity numeric(12,2) not null check (quantity > 0),
  usage_month date not null,
  status text not null default 'reserved' check (status in ('reserved', 'consumed', 'released')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  consumed_at timestamptz,
  released_at timestamptz,
  release_reason text
);

create index if not exists analysis_usage_reservations_workspace_month_status_idx
  on public.analysis_usage_reservations (workspace_id, usage_month, status);

alter table public.analysis_usage_reservations enable row level security;
revoke all on table public.analysis_usage_reservations from anon, authenticated;
grant all on table public.analysis_usage_reservations to service_role;

create or replace function public.reserve_analysis_minutes(
  p_workspace_id uuid,
  p_run_id uuid,
  p_user_id uuid,
  p_quantity numeric,
  p_allowance_minutes numeric
)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_month date := date_trunc('month', timezone('utc', now()))::date;
  v_used numeric := 0;
  v_reserved numeric := 0;
  v_existing public.analysis_usage_reservations%rowtype;
  v_remaining numeric;
begin
  if p_quantity is null or p_quantity <= 0 then
    raise exception 'p_quantity must be greater than zero';
  end if;

  perform pg_advisory_xact_lock(
    hashtextextended(p_workspace_id::text || ':' || v_month::text || ':analysis_minutes', 0)
  );

  select *
    into v_existing
    from public.analysis_usage_reservations
   where run_id = p_run_id;

  if found then
    if v_existing.workspace_id <> p_workspace_id then
      raise exception 'reservation workspace mismatch for run %', p_run_id;
    end if;

    if v_existing.status in ('reserved', 'consumed') then
      return jsonb_build_object(
        'granted', true,
        'idempotent', true,
        'status', v_existing.status,
        'reserved_minutes', v_existing.quantity
      );
    end if;

    return jsonb_build_object(
      'granted', false,
      'idempotent', true,
      'status', v_existing.status,
      'reason', 'reservation_released'
    );
  end if;

  select coalesce(sum(quantity), 0)
    into v_used
    from public.usage_events
   where workspace_id = p_workspace_id
     and metric_code = 'analysis_minutes'
     and occurred_at >= v_month::timestamptz
     and occurred_at < (v_month + interval '1 month')::timestamptz;

  select coalesce(sum(quantity), 0)
    into v_reserved
    from public.analysis_usage_reservations
   where workspace_id = p_workspace_id
     and usage_month = v_month
     and metric_code = 'analysis_minutes'
     and status = 'reserved';

  if p_allowance_minutes is not null
     and v_used + v_reserved + p_quantity > p_allowance_minutes + 0.001 then
    v_remaining := greatest(0, p_allowance_minutes - v_used - v_reserved);
    return jsonb_build_object(
      'granted', false,
      'used_minutes', v_used,
      'reserved_minutes', v_reserved,
      'remaining_minutes', v_remaining
    );
  end if;

  insert into public.analysis_usage_reservations (
    workspace_id,
    run_id,
    user_id,
    metric_code,
    quantity,
    usage_month
  ) values (
    p_workspace_id,
    p_run_id,
    p_user_id,
    'analysis_minutes',
    p_quantity,
    v_month
  );

  v_remaining := case
    when p_allowance_minutes is null then null
    else greatest(0, p_allowance_minutes - v_used - v_reserved - p_quantity)
  end;

  return jsonb_build_object(
    'granted', true,
    'used_minutes', v_used,
    'reserved_minutes', v_reserved + p_quantity,
    'remaining_minutes', v_remaining
  );
end;
$$;

create or replace function public.release_analysis_minutes_reservation(
  p_run_id uuid,
  p_reason text default null
)
returns boolean
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_reservation public.analysis_usage_reservations%rowtype;
begin
  select *
    into v_reservation
    from public.analysis_usage_reservations
   where run_id = p_run_id
   for update;

  if not found then
    return false;
  end if;

  if v_reservation.status = 'released' then
    return true;
  end if;

  if v_reservation.status <> 'reserved' then
    return false;
  end if;

  update public.analysis_usage_reservations
     set status = 'released',
         released_at = now(),
         release_reason = nullif(left(coalesce(p_reason, ''), 500), ''),
         updated_at = now()
   where id = v_reservation.id;

  return true;
end;
$$;

create or replace function public.consume_analysis_minutes_reservation(
  p_run_id uuid,
  p_actual_quantity numeric,
  p_metadata jsonb default '{}'::jsonb
)
returns boolean
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_reservation public.analysis_usage_reservations%rowtype;
  v_actual numeric;
begin
  select *
    into v_reservation
    from public.analysis_usage_reservations
   where run_id = p_run_id
   for update;

  if not found then
    return false;
  end if;

  if v_reservation.status = 'consumed' then
    return true;
  end if;

  if v_reservation.status <> 'reserved' then
    return false;
  end if;

  v_actual := least(
    v_reservation.quantity,
    greatest(0, coalesce(p_actual_quantity, 0))
  );

  if v_actual > 0 then
    insert into public.usage_events (
      workspace_id,
      user_id,
      analysis_run_id,
      metric_code,
      quantity,
      metadata
    ) values (
      v_reservation.workspace_id,
      v_reservation.user_id,
      v_reservation.run_id,
      'analysis_minutes',
      v_actual,
      coalesce(p_metadata, '{}'::jsonb) || jsonb_build_object(
        'reserved_minutes', v_reservation.quantity
      )
    );
  end if;

  update public.analysis_usage_reservations
     set status = 'consumed',
         consumed_at = now(),
         updated_at = now()
   where id = v_reservation.id;

  return true;
end;
$$;

revoke all on function public.reserve_analysis_minutes(uuid, uuid, uuid, numeric, numeric) from public, anon, authenticated;
revoke all on function public.release_analysis_minutes_reservation(uuid, text) from public, anon, authenticated;
revoke all on function public.consume_analysis_minutes_reservation(uuid, numeric, jsonb) from public, anon, authenticated;

grant execute on function public.reserve_analysis_minutes(uuid, uuid, uuid, numeric, numeric) to service_role;
grant execute on function public.release_analysis_minutes_reservation(uuid, text) to service_role;
grant execute on function public.consume_analysis_minutes_reservation(uuid, numeric, jsonb) to service_role;

commit;
