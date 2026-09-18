import Link from "next/link";
import { redirect } from "next/navigation";
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
  return (
    <>
      <AnalysisView runId={id} />
      <Link className={launcherStyles.launcher} href={`/analysis/${encodeURIComponent(id)}/review`}>
        Review workspace
      </Link>
      <AnalysisCancelControl runId={id} />
    </>
  );
}
