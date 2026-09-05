-- Lightweight team and roster management for roadmap #15.
-- Roster entries are current team context, not canonical cross-team player profiles.

alter table public.teams
  add column if not exists season text,
  add column if not exists age_group text,
  add column if not exists is_archived boolean not null default false,
  add column if not exists archived_at timestamptz;

create index if not exists teams_workspace_active_idx
  on public.teams(workspace_id, is_archived, name);

create table if not exists public.team_players (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  team_id uuid not null references public.teams(id) on delete cascade,
  name text not null,
  jersey_number smallint check (jersey_number is null or (jersey_number >= 0 and jersey_number <= 999)),
  position text,
  is_active boolean not null default true,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists team_players_workspace_idx on public.team_players(workspace_id);
create index if not exists team_players_team_active_idx on public.team_players(team_id, is_active, jersey_number, name);
create unique index if not exists team_players_active_jersey_unique
  on public.team_players(team_id, jersey_number)
  where is_active and jersey_number is not null;

create trigger team_players_set_updated_at
  before update on public.team_players
  for each row execute function public.set_updated_at();

alter table public.team_players enable row level security;

create policy team_players_select on public.team_players for select to authenticated
  using (public.is_workspace_member(workspace_id));
create policy team_players_insert on public.team_players for insert to authenticated
  with check (
    public.can_manage_workspace(workspace_id)
    and exists (
      select 1 from public.teams t
      where t.id = team_id and t.workspace_id = team_players.workspace_id
    )
  );
create policy team_players_update on public.team_players for update to authenticated
  using (public.can_manage_workspace(workspace_id))
  with check (
    public.can_manage_workspace(workspace_id)
    and exists (
      select 1 from public.teams t
      where t.id = team_id and t.workspace_id = team_players.workspace_id
    )
  );
create policy team_players_delete on public.team_players for delete to authenticated
  using (public.is_workspace_admin(workspace_id));

grant select, insert, update, delete on public.team_players to authenticated;
