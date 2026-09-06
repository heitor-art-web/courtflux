-- Add guided private-beta onboarding without disrupting existing workspaces.
-- Existing workspaces are marked complete once; new signups remain incomplete until
-- they finish role + first-team setup in the Next.js onboarding flow.

alter table public.profiles
  add column if not exists basketball_role text;

alter table public.workspaces
  add column if not exists primary_team_id uuid,
  add column if not exists onboarding_completed_at timestamptz;

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conname = 'profiles_basketball_role_check'
      and conrelid = 'public.profiles'::regclass
  ) then
    alter table public.profiles
      add constraint profiles_basketball_role_check
      check (basketball_role is null or basketball_role in ('coach','analyst','team_staff','academy_staff','other'));
  end if;
end;
$$;

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conname = 'workspaces_primary_team_id_fkey'
      and conrelid = 'public.workspaces'::regclass
  ) then
    alter table public.workspaces
      add constraint workspaces_primary_team_id_fkey
      foreign key (primary_team_id) references public.teams(id) on delete set null;
  end if;
end;
$$;

-- Do not interrupt users who already had a CourtFlux workspace before onboarding
-- shipped. New workspaces created after this migration retain the NULL default.
update public.workspaces
set onboarding_completed_at = now()
where onboarding_completed_at is null;

comment on column public.profiles.basketball_role is
  'Product-context role selected during onboarding; separate from workspace permission role.';
comment on column public.workspaces.primary_team_id is
  'First/primary basketball team selected during onboarding for workflow defaults.';
comment on column public.workspaces.onboarding_completed_at is
  'NULL for new workspaces until guided onboarding is completed.';
