import { redirect } from "next/navigation";
import { isP3FoundationEnabled } from "@/lib/p3/feature-flag";
import { isP4SpatialDiagnosticsEnabled } from "@/lib/p4/spatial-diagnostics-flag";
import { createClient } from "@/lib/supabase/server";
import { AnalysisCancelControl } from "./analysis-cancel-control";
import { AnalysisView } from "./analysis-view";
import { FoundationResultsPreview } from "./foundation-results-preview";

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
  const p3FoundationEnabled = isP3FoundationEnabled();
  const p4SpatialFoundationEnabled = isP4SpatialDiagnosticsEnabled();

  return (
    <>
      <AnalysisView runId={id} />
      {/*
        Keep the P3/P4 roadmap cards visible in the result experience even while
        their internal interactive surfaces remain feature-gated. This lets the
        private beta show what is coming without presenting unfinished outputs
        as if they were enabled or measured for the run.
      */}
      <FoundationResultsPreview
        runId={id}
        p3FoundationEnabled={p3FoundationEnabled}
        p4SpatialFoundationEnabled={p4SpatialFoundationEnabled}
      />
      <AnalysisCancelControl runId={id} />
    </>
  );
}
