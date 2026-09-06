revoke execute on function private.can_manage_workspace(uuid) from public, anon;
revoke execute on function private.is_workspace_admin(uuid) from public, anon;
revoke execute on function private.is_workspace_member(uuid) from public, anon;
revoke execute on function private.shares_workspace(uuid) from public, anon;
revoke execute on function private.workspace_role(uuid) from public, anon;
revoke execute on function private.is_workspace_owner(uuid) from public, anon;

grant execute on function private.can_manage_workspace(uuid) to authenticated;
grant execute on function private.is_workspace_admin(uuid) to authenticated;
grant execute on function private.is_workspace_member(uuid) to authenticated;
grant execute on function private.shares_workspace(uuid) to authenticated;
grant execute on function private.workspace_role(uuid) to authenticated;
grant execute on function private.is_workspace_owner(uuid) to authenticated;

drop function if exists public.handle_new_user();
