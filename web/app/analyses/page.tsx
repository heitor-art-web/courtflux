import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";
import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";
import { AnalysisLibrary, type LibraryAnalysis } from "./analysis-library";
import styles from "./page.module.css";

export const dynamic = "force-dynamic";

const MAX_LIBRARY_ITEMS = 500;

function durationMinutes(start: number | null, end: number | null) {
  if (typeof start !== "number" || typeof end !== "number" || end <= start) return null;
  return (end - start) / 60;
}

function sourceKind(value: string | null): LibraryAnalysis["sourceKind"] {
  if (!value) return "other";
  if (value.startsWith("supabase://analysis-videos/")) return "upload";
  if (/youtu(?:\.be|be\.com)/i.test(value)) return "youtube";
  return "other";
}

function uploadObjectPath(value: string | null) {
  const prefix = "supabase://analysis-videos/";
  return value?.startsWith(prefix) ? value.slice(prefix.length) : null;
}

function metadataSearchText(value: unknown) {
  if (!value || typeof value !== "object") return "";
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

export default async function AnalysesPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) redirect("/login");

  const { data: membership, error: membershipError } = await supabase
    .from("workspace_members")
    .select("workspace_id, role, workspaces(name)")
    .eq("user_id", user.id)
    .eq("status", "active")
    .order("created_at", { ascending: true })
    .limit(1)
    .maybeSingle();

  if (membershipError || !membership?.workspace_id) redirect("/account");

  const workspaceId = membership.workspace_id as string;
  const workspace = Array.isArray(membership.workspaces)
    ? membership.workspaces[0]
    : membership.workspaces;
  const admin = createAdminClient();

  const [{ data: runs, count, error: runsError }, { data: uploads, error: uploadsError }] = await Promise.all([
    supabase
      .from("analysis_runs")
      .select(
        "id, status, created_at, completed_at, start_time_sec, end_time_sec, video_source, games(id, title, competition, game_date, source_type, metadata)",
        { count: "exact" },
      )
      .eq("workspace_id", workspaceId)
      .order("created_at", { ascending: false })
      .limit(MAX_LIBRARY_ITEMS),
    admin
      .from("analysis_video_uploads")
      .select("object_path, expires_at, deleted_at")
      .eq("workspace_id", workspaceId),
  ]);

  if (runsError) throw runsError;

  const uploadMap = new Map(
    (uploadsError ? [] : uploads ?? []).map((upload) => [upload.object_path, upload]),
  );

  const analyses: LibraryAnalysis[] = (runs ?? []).map((run) => {
    const game = Array.isArray(run.games) ? run.games[0] : run.games;
    const kind = sourceKind(run.video_source);
    const objectPath = uploadObjectPath(run.video_source);
    const retention = objectPath ? uploadMap.get(objectPath) : null;
    const title = game?.title?.trim() || (kind === "youtube" ? "YouTube basketball analysis" : "Basketball analysis");

    return {
      id: run.id,
      status: run.status,
      title,
      competition: game?.competition ?? null,
      gameDate: game?.game_date ?? null,
      createdAt: run.created_at,
      completedAt: run.completed_at,
      durationMinutes: durationMinutes(run.start_time_sec, run.end_time_sec),
      sourceKind: kind,
      sourceExpiresAt: retention?.expires_at ?? null,
      sourceDeletedAt: retention?.deleted_at ?? null,
      sourceRetentionKnown: kind !== "upload" || Boolean(retention),
      searchText: [
        title,
        game?.competition ?? "",
        metadataSearchText(game?.metadata),
        kind,
      ].join(" ").toLowerCase(),
    };
  });

  const canStartAnalysis = ["owner", "admin", "analyst"].includes(membership.role ?? "");

  return (
    <main className={styles.page}>
      <header className={styles.topbar}>
        <Link href="/" className={styles.brand} aria-label="CourtFlux home">
          <Image
            src="/brand/horizontal_logo_safe_crop.png"
            alt="CourtFlux — Basketball Intelligence"
            width={660}
            height={185}
            priority
          />
        </Link>
        <nav className={styles.topActions} aria-label="Analysis library navigation">
          <Link href="/account">Dashboard</Link>
          {canStartAnalysis ? <Link className={styles.primaryNav} href="/analysis/new">New analysis</Link> : null}
        </nav>
      </header>

      <section className={styles.hero} aria-labelledby="library-title">
        <div>
          <div className={styles.eyebrow}>Analysis library</div>
          <h1 id="library-title">My analyses</h1>
          <p>
            Find every basketball analysis in {workspace?.name ?? "your CourtFlux workspace"} without having to remember whether it started as a full game, a shorter segment or a YouTube source.
          </p>
        </div>
        <div className={styles.heroNote}>
          <strong>Source video and analysis history are separate.</strong>
          <span>Uploaded source video is removed after 60 days. Results and statistics stay in your library.</span>
        </div>
      </section>

      <AnalysisLibrary
        analyses={analyses}
        totalCount={count ?? analyses.length}
        canStartAnalysis={canStartAnalysis}
        maxItems={MAX_LIBRARY_ITEMS}
      />
    </main>
  );
}
