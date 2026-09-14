import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { AnalysisCancelControl } from "./analysis-cancel-control";
import { AnalysisView } from "./analysis-view";

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
      <AnalysisCancelControl runId={id} />
    </>
  );
}
