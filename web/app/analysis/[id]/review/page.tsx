import Link from "next/link";
import { redirect } from "next/navigation";
import { createClient } from "@/lib/supabase/server";
import { ReviewWorkspace } from "./review-workspace";
import styles from "./review-page.module.css";

export const dynamic = "force-dynamic";

export default async function ReviewPage({
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
    <main className={styles.page}>
      <nav className={styles.topbar} aria-label="Review workspace navigation">
        <Link href={`/analysis/${encodeURIComponent(id)}`}>← Back to analysis</Link>
        <Link href="/account">Dashboard</Link>
      </nav>
      <ReviewWorkspace runId={id} />
    </main>
  );
}
