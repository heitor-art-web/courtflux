import Link from "next/link";
import { redirect } from "next/navigation";
import { SignOutButton } from "./signout-button";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

export default async function AccountPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) redirect("/login");

  const { data: membership } = await supabase
    .from("workspace_members")
    .select("workspace_id, role, workspaces(name, slug)")
    .eq("user_id", user.id)
    .eq("status", "active")
    .limit(1)
    .maybeSingle();

  const workspaceId = membership?.workspace_id ?? null;
  const { data: entitlement } = workspaceId
    ? await supabase
        .from("workspace_entitlements")
        .select("plan_code, plan_name, status, current_period_end, cancel_at_period_end, features, limits")
        .eq("workspace_id", workspaceId)
        .maybeSingle()
    : { data: null };

  const workspace = Array.isArray(membership?.workspaces)
    ? membership?.workspaces[0]
    : membership?.workspaces;

  return (
    <main style={{ maxWidth: 900, margin: "0 auto" }}>
      <header>
        <div>
          <div className="muted" style={{ letterSpacing: "0.12em", textTransform: "uppercase" }}>CourtFlux account</div>
          <h1 style={{ marginBottom: 8 }}>{user.email}</h1>
          <p className="muted">Workspace, access and subscription.</p>
        </div>
        <div className="actions">
          <Link href="/">Back to analysis</Link>
          <SignOutButton />
        </div>
      </header>

      <div className="grid">
        <section className="panel span-6">
          <h2>Workspace</h2>
          <div className="metrics" style={{ marginTop: 16 }}>
            <div className="metric"><div className="label">Name</div><div className="value" style={{ fontSize: 20 }}>{workspace?.name ?? "—"}</div></div>
            <div className="metric"><div className="label">Role</div><div className="value" style={{ fontSize: 20 }}>{membership?.role ?? "—"}</div></div>
          </div>
        </section>

        <section className="panel span-6">
          <h2>Subscription</h2>
          <div className="metrics" style={{ marginTop: 16 }}>
            <div className="metric"><div className="label">Plan</div><div className="value" style={{ fontSize: 24 }}>{entitlement?.plan_name ?? "Free"}</div></div>
            <div className="metric"><div className="label">Status</div><div className="value" style={{ fontSize: 20 }}>{entitlement?.status ?? "free"}</div></div>
          </div>
          <p className="muted" style={{ marginTop: 16 }}>
            Free, Base and Premium are represented in the database. Checkout and provider webhooks will attach to this subscription record without changing the account model.
          </p>
        </section>

        <section className="panel span-12">
          <h2>Entitlements</h2>
          <pre>{JSON.stringify({ features: entitlement?.features ?? {}, limits: entitlement?.limits ?? {} }, null, 2)}</pre>
        </section>
      </div>
    </main>
  );
}
