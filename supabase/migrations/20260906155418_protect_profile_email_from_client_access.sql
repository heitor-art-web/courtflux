revoke select, update on table public.profiles from anon, authenticated;

grant select (id, full_name, avatar_url, basketball_role, created_at, updated_at)
  on table public.profiles to authenticated;

grant update (full_name, avatar_url, basketball_role)
  on table public.profiles to authenticated;
