-- P3 foundation: private durable storage for generated analysis artifacts.
--
-- P3 physical clips are worker-produced and browser clients never upload them directly.
-- The web API validates workspace/run ownership and issues short-lived signed URLs only
-- for artifacts already linked to a workspace-scoped p3_clips row.

insert into storage.buckets (
  id,
  name,
  public,
  file_size_limit,
  allowed_mime_types
)
values (
  'analysis-artifacts',
  'analysis-artifacts',
  false,
  536870912,
  array[
    'video/mp4',
    'application/json',
    'image/jpeg',
    'image/png'
  ]
)
on conflict (id) do update set
  name = excluded.name,
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

comment on column public.p3_clips.artifact_id is
  'Durable analysis_artifacts row for a ready physical clip. NULL while planned/generating/failed.';

comment on column public.analysis_artifacts.checksum is
  'Content checksum when known. P3 evidence clips use SHA-256 and content-addressed Storage paths.';
