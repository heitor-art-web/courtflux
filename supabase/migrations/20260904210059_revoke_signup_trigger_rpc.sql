-- Replacing the signup trigger function restores PostgreSQL's default PUBLIC
-- EXECUTE privilege. Keep it trigger-only so it cannot be called through the
-- Data API as a security-definer RPC.
revoke execute on function public.handle_new_user() from public, anon, authenticated;
