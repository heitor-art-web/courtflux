import type { Metadata } from "next";
import { redirect } from "next/navigation";
import mobileStyles from "../product-mobile.module.css";
import { createClient } from "@/lib/supabase/server";

export const metadata: Metadata = {
  title: "Teams & Rosters",
  description: "Manage private CourtFlux teams and current roster context.",
  robots: {
    index: false,
    follow: false,
    nocache: true,
  },
};

export default async function TeamsLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) redirect("/login");

  const { data: membership } = await supabase
    .from("workspace_members")
    .select("workspace_id, workspaces(onboarding_completed_at)")
    .eq("user_id", user.id)
    .eq("status", "active")
    .order("created_at", { ascending: true })
    .limit(1)
    .maybeSingle();

  const workspace = Array.isArray(membership?.workspaces)
    ? membership.workspaces[0]
    : membership?.workspaces;

  if (workspace && !workspace.onboarding_completed_at) redirect("/onboarding");

  return <div className={mobileStyles.productViewport}>{children}</div>;
}
