import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { OnboardingFlow } from "./onboarding-flow";

export const dynamic = "force-dynamic";

export default async function OnboardingPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) redirect("/login?mode=signup");

  const [{ data: profile }, { data: membership }] = await Promise.all([
    supabase
      .from("profiles")
      .select("full_name, basketball_role")
      .eq("id", user.id)
      .maybeSingle(),
    supabase
      .from("workspace_members")
      .select("workspace_id, role, workspaces(name, onboarding_completed_at)")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle(),
  ]);

  const workspace = Array.isArray(membership?.workspaces)
    ? membership.workspaces[0]
    : membership?.workspaces;

  if (!membership?.workspace_id || !workspace) redirect("/account");
  if (workspace.onboarding_completed_at) redirect("/account");

  const displayName = profile?.full_name?.trim() || user.email?.split("@")[0] || "Coach";

  return (
    <OnboardingFlow
      displayName={displayName}
      defaultWorkspaceName={workspace.name || `${displayName} Workspace`}
      initialRole={profile?.basketball_role ?? null}
    />
  );
}
