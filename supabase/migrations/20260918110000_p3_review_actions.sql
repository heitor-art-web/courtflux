-- P3.7 review actions: personal organization over durable P3 review items.
-- These tables deliberately reference p3_review_items instead of copying event truth.
-- If P2/P3 is reprocessed, historical review items remain auditable and user actions
-- stay separate from model/source predictions.

create table if not exists public.p3_saved_items (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  review_item_id uuid not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),

  unique (user_id, review_item_id),
  foreign key (review_item_id, workspace_id)
    references public.p3_review_items(id, workspace_id) on delete cascade
);

create index if not exists p3_saved_items_workspace_user_idx
  on public.p3_saved_items(workspace_id, user_id, created_at desc);

alter table public.p3_saved_items enable row level security;
create policy p3_saved_items_select
  on public.p3_saved_items for select to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_saved_items_insert
  on public.p3_saved_items for insert to authenticated
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_saved_items_delete
  on public.p3_saved_items for delete to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));


create table if not exists public.p3_tags (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  normalized_name text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  unique (id, workspace_id, user_id),
  unique (workspace_id, user_id, normalized_name),
  check (length(trim(name)) between 1 and 64),
  check (length(trim(normalized_name)) between 1 and 64)
);

create index if not exists p3_tags_workspace_user_idx
  on public.p3_tags(workspace_id, user_id, normalized_name);

alter table public.p3_tags enable row level security;
create policy p3_tags_select
  on public.p3_tags for select to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_tags_insert
  on public.p3_tags for insert to authenticated
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_tags_update
  on public.p3_tags for update to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id))
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_tags_delete
  on public.p3_tags for delete to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));

create trigger p3_tags_set_updated_at
  before update on public.p3_tags
  for each row execute function private.set_updated_at();


create table if not exists public.p3_review_item_tags (
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  review_item_id uuid not null,
  tag_id uuid not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),

  primary key (review_item_id, tag_id, user_id),
  foreign key (review_item_id, workspace_id)
    references public.p3_review_items(id, workspace_id) on delete cascade,
  foreign key (tag_id, workspace_id, user_id)
    references public.p3_tags(id, workspace_id, user_id) on delete cascade
);

create index if not exists p3_review_item_tags_user_idx
  on public.p3_review_item_tags(workspace_id, user_id, created_at desc);

alter table public.p3_review_item_tags enable row level security;
create policy p3_review_item_tags_select
  on public.p3_review_item_tags for select to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_review_item_tags_insert
  on public.p3_review_item_tags for insert to authenticated
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_review_item_tags_delete
  on public.p3_review_item_tags for delete to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));


create table if not exists public.p3_notes (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  review_item_id uuid not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  body text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  unique (user_id, review_item_id),
  foreign key (review_item_id, workspace_id)
    references public.p3_review_items(id, workspace_id) on delete cascade,
  check (length(trim(body)) between 1 and 4000)
);

create index if not exists p3_notes_workspace_user_idx
  on public.p3_notes(workspace_id, user_id, updated_at desc);

alter table public.p3_notes enable row level security;
create policy p3_notes_select
  on public.p3_notes for select to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_notes_insert
  on public.p3_notes for insert to authenticated
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_notes_update
  on public.p3_notes for update to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id))
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_notes_delete
  on public.p3_notes for delete to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));

create trigger p3_notes_set_updated_at
  before update on public.p3_notes
  for each row execute function private.set_updated_at();


create table if not exists public.p3_collections (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  description text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  unique (id, workspace_id, user_id),
  check (length(trim(name)) between 1 and 120),
  check (description is null or length(description) <= 1000)
);

create index if not exists p3_collections_workspace_user_idx
  on public.p3_collections(workspace_id, user_id, updated_at desc);

alter table public.p3_collections enable row level security;
create policy p3_collections_select
  on public.p3_collections for select to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_collections_insert
  on public.p3_collections for insert to authenticated
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_collections_update
  on public.p3_collections for update to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id))
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_collections_delete
  on public.p3_collections for delete to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));

create trigger p3_collections_set_updated_at
  before update on public.p3_collections
  for each row execute function private.set_updated_at();


create table if not exists public.p3_collection_items (
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  collection_id uuid not null,
  review_item_id uuid not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  position integer not null default 0 check (position >= 0),
  created_at timestamptz not null default now(),

  primary key (collection_id, review_item_id),
  foreign key (collection_id, workspace_id, user_id)
    references public.p3_collections(id, workspace_id, user_id) on delete cascade,
  foreign key (review_item_id, workspace_id)
    references public.p3_review_items(id, workspace_id) on delete cascade
);

create index if not exists p3_collection_items_user_idx
  on public.p3_collection_items(workspace_id, user_id, collection_id, position, created_at);

alter table public.p3_collection_items enable row level security;
create policy p3_collection_items_select
  on public.p3_collection_items for select to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_collection_items_insert
  on public.p3_collection_items for insert to authenticated
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_collection_items_update
  on public.p3_collection_items for update to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id))
  with check (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));
create policy p3_collection_items_delete
  on public.p3_collection_items for delete to authenticated
  using (user_id = (select auth.uid()) and private.is_workspace_member(workspace_id));


revoke all on public.p3_saved_items from public, anon, authenticated;
revoke all on public.p3_tags from public, anon, authenticated;
revoke all on public.p3_review_item_tags from public, anon, authenticated;
revoke all on public.p3_notes from public, anon, authenticated;
revoke all on public.p3_collections from public, anon, authenticated;
revoke all on public.p3_collection_items from public, anon, authenticated;

grant select, insert, delete on public.p3_saved_items to authenticated;
grant select, insert, update, delete on public.p3_tags to authenticated;
grant select, insert, delete on public.p3_review_item_tags to authenticated;
grant select, insert, update, delete on public.p3_notes to authenticated;
grant select, insert, update, delete on public.p3_collections to authenticated;
grant select, insert, update, delete on public.p3_collection_items to authenticated;

grant select, insert, update, delete on public.p3_saved_items to service_role;
grant select, insert, update, delete on public.p3_tags to service_role;
grant select, insert, update, delete on public.p3_review_item_tags to service_role;
grant select, insert, update, delete on public.p3_notes to service_role;
grant select, insert, update, delete on public.p3_collections to service_role;
grant select, insert, update, delete on public.p3_collection_items to service_role;

comment on table public.p3_saved_items is
  'Personal P3.7 saved-state for durable review items. Does not copy or mutate event truth.';
comment on table public.p3_tags is
  'Personal workspace-scoped P3.7 tags used to organize review items.';
comment on table public.p3_notes is
  'Personal P3.7 note attached to a durable review item; model output remains immutable.';
comment on table public.p3_collections is
  'Personal P3.7 review collections. Collaboration/sharing semantics are intentionally out of scope.';
