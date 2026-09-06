-- Private source-video storage for the authenticated New Analysis flow.
-- Uploads are issued through short-lived signed upload tokens by the Next.js server;
-- the browser never receives the service-role key and objects are never public.
insert into storage.buckets (
  id,
  name,
  public,
  file_size_limit,
  allowed_mime_types
)
values (
  'analysis-videos',
  'analysis-videos',
  false,
  2147483648,
  array[
    'video/mp4',
    'video/quicktime',
    'video/webm',
    'video/x-m4v'
  ]
)
on conflict (id) do update set
  name = excluded.name,
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;
