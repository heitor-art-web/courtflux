import Link from "next/link";
import { redirect } from "next/navigation";
import { isP4SpatialDiagnosticsEnabled } from "@/lib/p4/spatial-diagnostics-flag";
import { parseSpatialSnapshotRows, type SpatialSnapshotRow } from "@/lib/p4/spatial-viewer";
import { createClient } from "@/lib/supabase/server";
import { SpatialRadar, type SpatialRunMeta } from "./spatial-radar";
import styles from "./spatial-radar.module.css";

export const dynamic = "force-dynamic";

export default async function SpatialDiagnosticsPage({
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
  if (!isP4SpatialDiagnosticsEnabled()) {
    redirect(`/analysis/${encodeURIComponent(id)}`);
  }

  let run: SpatialRunMeta | null = null;
  let frames = [] as ReturnType<typeof parseSpatialSnapshotRows>["frames"];
  let rejectedRows = 0;
  let dataError: string | null = null;

  const { data: runData, error: runError } = await supabase
    .from("p4_spatial_runs")
    .select("id,status,contract_version,spatial_runtime_version,created_at,completed_at")
    .eq("analysis_run_id", id)
    .in("status", ["current", "building"])
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (runError) {
    dataError = "P4 spatial persistence is unavailable in this environment. Apply the source-controlled P4 migration before using the internal viewer.";
  } else if (runData) {
    run = {
      id: String(runData.id),
      status: String(runData.status),
      contractVersion: String(runData.contract_version),
      spatialRuntimeVersion: String(runData.spatial_runtime_version),
      createdAt: runData.created_at ? String(runData.created_at) : null,
      completedAt: runData.completed_at ? String(runData.completed_at) : null,
    };

    const { data: snapshotRows, error: snapshotError } = await supabase
      .from("p4_spatial_snapshots")
      .select("payload")
      .eq("spatial_run_id", run.id)
      .order("timestamp_ms", { ascending: true })
      .order("frame_id", { ascending: true })
      .limit(2000);

    if (snapshotError) {
      dataError = "Materialized P4 spatial snapshots could not be read for this analysis.";
    } else {
      const parsed = parseSpatialSnapshotRows((snapshotRows ?? []) as SpatialSnapshotRow[]);
      frames = parsed.frames;
      rejectedRows = parsed.rejected;
    }
  }

  return (
    <main className={styles.page}>
      <nav className={styles.topbar} aria-label="Spatial diagnostics navigation">
        <Link href={`/analysis/${encodeURIComponent(id)}`}>← Back to analysis</Link>
        <span>Internal · P4 Spatial Foundation</span>
      </nav>
      <SpatialRadar run={run} frames={frames} rejectedRows={rejectedRows} dataError={dataError} />
    </main>
  );
}
