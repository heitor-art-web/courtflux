import Link from "next/link";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { ReviewWorkspace } from "./review-workspace";
import type { ViewMode } from "./review-workspace-types";
import styles from "./review-page.module.css";

export const dynamic = "force-dynamic";

function textParam(value: string | string[] | undefined) {
  if (typeof value === "string" && value.trim()) return value.trim();
  if (Array.isArray(value) && typeof value[0] === "string" && value[0].trim()) return value[0].trim();
  return null;
}

function initialViewFromSearch(search: Record<string, string | string[] | undefined>): ViewMode {
  const view = textParam(search.view);
  if (view === "reviewed") return { kind: "reviewed" };
  if (view === "unreviewed") return { kind: "unreviewed" };
  if (view === "saved") return { kind: "saved" };
  if (view === "collection") {
    const collectionId = textParam(search.collection);
    if (collectionId) return { kind: "collection", collectionId };
  }
  if (view === "tag") {
    const tagId = textParam(search.tag);
    if (tagId) return { kind: "tag", tagId };
  }
  return { kind: "all" };
}

export default async function ReviewPage({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login");

  const [{ id }, search] = await Promise.all([params, searchParams]);
  return (
    <main className={styles.page}>
      <nav className={styles.topbar} aria-label="Review workspace navigation">
        <Link href={`/analysis/${encodeURIComponent(id)}`}>← Back to analysis</Link>
        <Link href="/account">Dashboard</Link>
      </nav>
      <ReviewWorkspace
        runId={id}
        initialDerivationId={textParam(search.derivation)}
        initialView={initialViewFromSearch(search)}
      />
    </main>
  );
}
