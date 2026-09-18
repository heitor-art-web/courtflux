import Link from "next/link";
import { redirect } from "next/navigation";
import { isP3FoundationEnabled } from "@/lib/p3/feature-flag";
import { createClient } from "@/lib/supabase/server";
import { AnalysisCancelControl } from "./analysis-cancel-control";
import { AnalysisView } from "./analysis-view";
import launcherStyles from "./review-workspace-launcher.module.css";

export const dynamic = "force-dynamic";

export default async function AnalysisPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) redirect("/login");

  const { id } = await params;
  const p3Enabled = isP3FoundationEnabled();

  return (
    <>
      <AnalysisView runId={id} />
      {p3Enabled ? (
        <Link className={launcherStyles.launcher} href={`/analysis/${encodeURIComponent(id)}/review`}>
          Review workspace
        </Link>
      ) : null}
      <AnalysisCancelControl runId={id} />
    </>
  );
}
