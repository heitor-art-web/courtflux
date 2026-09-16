begin;

-- Some browsers report otherwise valid MP4/MOV/WEBM/M4V files as
-- application/octet-stream. The upload-ticket API already validates the
-- filename extension and file size before issuing a signed token, so keep the
-- Storage bucket MIME contract aligned with that accepted browser behavior.
update storage.buckets
set allowed_mime_types = array[
  'video/mp4',
  'video/quicktime',
  'video/webm',
  'video/x-m4v',
  'application/octet-stream'
]
where id = 'analysis-videos';

commit;
