-- P3 foundation: evidence clips, derived review items, and human corrections.
-- Model/source output stays worker-owned; human feedback is stored separately so an
-- original prediction can always be reconstructed and evaluated later.

create table if not exists public.p3_clips (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid not null references public.games(id) on delete cascade,
  analysis_run_id uuid not null references public.analysis_runs(id) on delete cascade,
  derivation_id uuid not null references public.analysis_derivations(id) on delete cascade,
  artifact_id uuid references public.analysis_artifacts(id) on delete set null,

  review_type text not null,
  source_start_ms bigint not null check (source_start_ms >= 0),
  source_end_ms bigint not null check (source_end_ms >= source_start_ms),
  clip_start_ms bigint not null check (clip_start_ms >= 0),
  clip_end_ms bigint not null check (clip_end_ms >= clip_start_ms),
  source_event_ids text[] not null default '{}'::text[],
  game_player_ids text[] not null default '{}'::text[],

  context_policy_version text not null,
  generation_version text not null,
  status text not null default 'planned'
    check (status in ('planned','generating','ready','failed')),
  error text,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  check (length(trim(review_type)) > 0),
  check (length(trim(context_policy_version)) > 0),
  check (length(trim(generation_version)) > 0)
);

create unique index if not exists p3_clips_identity_idx
  on public.p3_clips(derivation_id, review_type, clip_start_ms, clip_end_ms, source_event_ids);
create index if not exists p3_clips_game_status_idx
  on public.p3_clips(game_id, status, created_at desc);

alter table public.p3_clips enable row level security;
create policy p3_clips_select
  on public.p3_clips
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));

revoke all on public.p3_clips from public, anon, authenticated;
grant select on public.p3_clips to authenticated;
grant select, insert, update, delete on public.p3_clips to service_role;

create trigger p3_clips_set_updated_at
  before update on public.p3_clips
  for each row execute function private.set_updated_at();


create table if not exists public.p3_review_items (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  game_id uuid not null references public.games(id) on delete cascade,
  analysis_run_id uuid not null references public.analysis_runs(id) on delete cascade,
  derivation_id uuid not null references public.analysis_derivations(id) on delete cascade,
  clip_id uuid references public.p3_clips(id) on delete set null,

  review_type text not null,
  source_event_ids text[] not null default '{}'::text[],
  source_possession_id text,
  game_player_ids text[] not null default '{}'::text[],
  team_id text,

  confidence double precision check (confidence is null or (confidence >= 0 and confidence <= 1)),
  support_state text not null,
  completeness text not null default 'unknown',
  reason_shown jsonb not null default '{}'::jsonb,

  created_at timestamptz not null default now(),

  unique (id, workspace_id),
  check (length(trim(review_type)) > 0),
  check (length(trim(support_state)) > 0),
  check (jsonb_typeof(reason_shown) = 'object')
);

create index if not exists p3_review_items_game_type_idx
  on public.p3_review_items(game_id, review_type, created_at desc);
create index if not exists p3_review_items_derivation_idx
  on public.p3_review_items(derivation_id, created_at);

alter table public.p3_review_items enable row level security;
create policy p3_review_items_select
  on public.p3_review_items
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));

revoke all on public.p3_review_items from public, anon, authenticated;
grant select on public.p3_review_items to authenticated;
grant select, insert, update, delete on public.p3_review_items to service_role;


create table if not exists public.p3_reviews (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  review_item_id uuid not null,
  reviewer_id uuid not null references auth.users(id) on delete cascade,

  review_state text not null
    check (review_state in ('confirmed','corrected','rejected')),
  corrected_event_type text
    check (
      corrected_event_type is null
      or corrected_event_type in (
        'pass','turnover','shot','rebound','loose_ball','dunk','alley_oop','none'
      )
    ),
  corrected_game_player_id text,
  note text,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  unique (review_item_id, reviewer_id),
  foreign key (review_item_id, workspace_id)
    references public.p3_review_items(id, workspace_id) on delete cascade,

  check (
    (review_state = 'confirmed'
      and corrected_event_type is null
      and corrected_game_player_id is null)
    or
    (review_state = 'corrected'
      and corrected_event_type is distinct from 'none'
      and (corrected_event_type is not null or corrected_game_player_id is not null))
    or
    (review_state = 'rejected'
      and corrected_event_type = 'none'
      and corrected_game_player_id is null)
  )
);

create index if not exists p3_reviews_workspace_time_idx
  on public.p3_reviews(workspace_id, updated_at desc);
create index if not exists p3_reviews_reviewer_idx
  on public.p3_reviews(reviewer_id, updated_at desc);

alter table public.p3_reviews enable row level security;

create policy p3_reviews_select
  on public.p3_reviews
  for select
  to authenticated
  using (private.is_workspace_member(workspace_id));

create policy p3_reviews_insert
  on public.p3_reviews
  for insert
  to authenticated
  with check (
    reviewer_id = (select auth.uid())
    and private.can_manage_workspace(workspace_id)
  );

create policy p3_reviews_update
  on public.p3_reviews
  for update
  to authenticated
  using (
    reviewer_id = (select auth.uid())
    and private.can_manage_workspace(workspace_id)
  )
  with check (
    reviewer_id = (select auth.uid())
    and private.can_manage_workspace(workspace_id)
  );

revoke all on public.p3_reviews from public, anon, authenticated;
grant select, insert, update on public.p3_reviews to authenticated;
grant select, insert, update, delete on public.p3_reviews to service_role;

create trigger p3_reviews_set_updated_at
  before update on public.p3_reviews
  for each row execute function private.set_updated_at();

comment on table public.p3_clips is
  'Regenerable P3 evidence-clip requests/status linked to source-video time and optional analysis_artifacts storage.';
comment on table public.p3_review_items is
  'Worker-derived P3 review items. Browser clients read them but do not mutate model/source truth.';
comment on table public.p3_reviews is
  'Human review/correction state stored separately from immutable P3 source and review-item predictions.';
