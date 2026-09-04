drop policy if exists analysis_video_uploads_deny_client_access on public.analysis_video_uploads;

create policy analysis_video_uploads_deny_client_access
on public.analysis_video_uploads
as restrictive
for all
to authenticated
using (false)
with check (false);
