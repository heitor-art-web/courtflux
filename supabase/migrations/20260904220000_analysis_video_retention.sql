create table if not exists public.analysis_video_uploads (
  id uuid primary key default gen_random_uuid(),
  workspace_id uuid not null references public.workspaces(id) on delete cascade,
  created_by uuid,
  bucket text not null default 'analysis-videos',
  object_path text not null unique,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null default (now() + interval '60 days'),
  purge_attempted_at timestamptz,
  deleted_at timestamptz,
  deletion_error text,
  constraint analysis_video_uploads_bucket_check check (bucket = 'analysis-videos')
);

create index if not exists analysis_video_uploads_expiry_idx
  on public.analysis_video_uploads (expires_at)
  where deleted_at is null;

create index if not exists analysis_video_uploads_workspace_idx
  on public.analysis_video_uploads (workspace_id, created_at desc);

alter table public.analysis_video_uploads enable row level security;
revoke all on table public.analysis_video_uploads from anon, authenticated;
grant select, insert, update, delete on table public.analysis_video_uploads to service_role;

-- Bring any existing private uploads under the same 60-day policy.
insert into public.analysis_video_uploads (
  workspace_id,
  created_by,
  bucket,
  object_path,
  created_at,
  expires_at
)
select
  w.id,
  case
    when split_part(o.name, '/', 2) ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
      then split_part(o.name, '/', 2)::uuid
    else null
  end,
  o.bucket_id,
  o.name,
  o.created_at,
  o.created_at + interval '60 days'
from storage.objects o
join public.workspaces w
  on split_part(o.name, '/', 1) = w.id::text
where o.bucket_id = 'analysis-videos'
  and o.name ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/'
on conflict (object_path) do nothing;
