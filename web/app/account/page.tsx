import Image from "next/image";
import Link from "next/link";
import { redirect } from "next/navigation";
import { SignOutButton } from "./signout-button";
import styles from "./page.module.css";
import { createClient } from "@/lib/supabase/server";
import {
  canStartAnalysis,
  getUsageLevel,
  getUsagePercent,
  normalizePlanCode,
  resolveMonthlyAllowance,
  USAGE_WARNING_PERCENT,
} from "@/lib/usage";

export const dynamic = "force-dynamic";

const PLAN_NAMES: Record<string, string> = {
  beta: "Beta",
  coach: "Coach",
  team: "Team",
  academy: "Academy",
};

function formatMinutes(value: number) {
  if (!Number.isFinite(value)) return "0";
  if (value < 10) return value.toFixed(1).replace(/\.0$/, "");
  return Math.round(value).toString();
}

function formatDate(value: string | null | undefined) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

function formatDuration(start: number | null | undefined, end: number | null | undefined) {
  if (typeof start !== "number" || typeof end !== "number" || end <= start) return null;
  return `${formatMinutes((end - start) / 60)} min`;
}

function sourceLabel(value: string | null | undefined) {
  if (!value) return "Video analysis";
  if (value.startsWith("supabase://")) return "Uploaded video";
  if (/youtu(?:\.be|be\.com)/i.test(value)) return "YouTube video";
  const clean = value.split(/[?#]/)[0];
  const segment = clean.split(/[\\/]/).filter(Boolean).at(-1) ?? value;
  return segment.length > 42 ? `${segment.slice(0, 39)}…` : segment;
}

function numberFromUnknown(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function statusClass(status: string) {
  const classes: Record<string, string> = {
    completed: styles.statusCompleted,
    running: styles.statusRunning,
    queued: styles.statusQueued,
    failed: styles.statusFailed,
    canceled: styles.statusCanceled,
  };
  return `${styles.status} ${classes[status] ?? ""}`;
}

export default async function AccountPage() {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) redirect("/login");

  const [{ data: profile }, { data: membership }] = await Promise.all([
    supabase.from("profiles").select("full_name").eq("id", user.id).maybeSingle(),
    supabase
      .from("workspace_members")
      .select("workspace_id, role, workspaces(name, slug)")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle(),
  ]);

  const workspaceId = membership?.workspace_id ?? null;
  const workspace = Array.isArray(membership?.workspaces)
    ? membership.workspaces[0]
    : membership?.workspaces;

  const now = new Date();
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1));
  const nextMonth = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() + 1, 1));

  const [
    entitlementResult,
    usageResult,
    grantsResult,
    runsResult,
    activeResult,
    analysesCountResult,
    teamsResult,
  ] = workspaceId
    ? await Promise.all([
        supabase
          .from("workspace_entitlements")
          .select("plan_code, plan_name, status, current_period_end, cancel_at_period_end, features, limits")
          .eq("workspace_id", workspaceId)
          .maybeSingle(),
        supabase
          .from("workspace_usage_monthly")
          .select("quantity")
          .eq("workspace_id", workspaceId)
          .eq("metric_code", "analysis_minutes")
          .gte("usage_month", monthStart.toISOString())
          .lt("usage_month", nextMonth.toISOString())
          .maybeSingle(),
        supabase
          .from("usage_allowance_grants")
          .select("quantity, valid_from, valid_until")
          .eq("workspace_id", workspaceId)
          .eq("metric_code", "analysis_minutes")
          .order("created_at", { ascending: false })
          .limit(100),
        supabase
          .from("analysis_runs")
          .select("id, status, created_at, completed_at, start_time_sec, end_time_sec, video_source, games(title, game_date)")
          .eq("workspace_id", workspaceId)
          .order("created_at", { ascending: false })
          .limit(6),
        supabase
          .from("analysis_runs")
          .select("id", { count: "exact", head: true })
          .eq("workspace_id", workspaceId)
          .in("status", ["queued", "running"]),
        supabase
          .from("analysis_runs")
          .select("id", { count: "exact", head: true })
          .eq("workspace_id", workspaceId),
        supabase
          .from("teams")
          .select("id, name, short_name", { count: "exact" })
          .eq("workspace_id", workspaceId)
          .order("name")
          .limit(6),
      ])
    : [
        { data: null },
        { data: null },
        { data: [] },
        { data: [] },
        { count: 0 },
        { count: 0 },
        { data: [], count: 0 },
      ];

  const entitlement = entitlementResult.data;
  const recentRuns = runsResult.data ?? [];
  const teams = teamsResult.data ?? [];
  const usedMinutes = numberFromUnknown(usageResult.data?.quantity) ?? 0;
  const limits = (entitlement?.limits ?? {}) as Record<string, unknown>;
  const nowMs = now.getTime();
  const grantedMinutes = (grantsResult.data ?? []).reduce((total, grant) => {
    const startsAt = new Date(grant.valid_from).getTime();
    const endsAt = grant.valid_until ? new Date(grant.valid_until).getTime() : null;
    const active = Number.isFinite(startsAt) && startsAt <= nowMs && (endsAt === null || endsAt > nowMs);
    return active ? total + (numberFromUnknown(grant.quantity) ?? 0) : total;
  }, 0);

  const planCode = normalizePlanCode(entitlement?.plan_code);
  const planName = PLAN_NAMES[planCode] ?? entitlement?.plan_name ?? "Beta";
  const baseMonthlyLimit = resolveMonthlyAllowance(planCode, limits, 0);
  const monthlyLimit = resolveMonthlyAllowance(planCode, limits, grantedMinutes);
  const remainingMinutes = monthlyLimit === null ? null : Math.max(0, monthlyLimit - usedMinutes);
  const usagePercent = getUsagePercent(usedMinutes, monthlyLimit);
  const usageLevel = getUsageLevel(usedMinutes, monthlyLimit);
  const analysisAllowed = canStartAnalysis(usedMinutes, monthlyLimit);
  const displayName = profile?.full_name?.trim() || user.email?.split("@")[0] || "Coach";
  const firstName = displayName.split(/\s+/)[0];
  const role = membership?.role ?? "member";
  const canManageAnalysis = analysisAllowed && ["owner", "admin", "analyst"].includes(role);

  return (
    <main className={styles.dashboard}>
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
        <div className={styles.topActions}>
          <Link href="/basketball-video-analysis">How analysis works</Link>
          <Link href="/">Public site</Link>
          <SignOutButton />
        </div>
      </header>

      <section className={styles.hero} aria-labelledby="dashboard-title">
        <div>
          <div className={styles.eyebrow}>CourtFlux workspace</div>
          <h1 id="dashboard-title">Welcome back, <span>{firstName}.</span></h1>
          <p>
            Review your monthly analysis usage and recent processing activity, then start a new customer analysis without opening engineering controls.
          </p>
        </div>
        <div className={styles.primaryAction}>
          <strong>{canManageAnalysis ? "Analyse basketball video" : analysisAllowed ? "Analysis access is view-only" : "Monthly allowance used"}</strong>
          <span>
            {canManageAnalysis
              ? "Upload game film or add a YouTube source, enter the two teams and review the minute estimate before processing."
              : analysisAllowed
                ? "Your workspace role can review analyses but cannot start a new one."
                : "New analyses pause at 100% of the monthly allowance. Ask the CourtFlux team for a beta top-up or compare plans."}
          </span>
          {canManageAnalysis ? (
            <Link className={styles.primaryButton} href="/analysis/new">Start new analysis</Link>
          ) : (
            <span className={styles.disabledButton} aria-disabled="true">Analysis unavailable</span>
          )}
        </div>
      </section>

      <section className={styles.metrics} aria-label="Workspace overview">
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Analysis minutes used</div>
          <div className={styles.metricValue}>{formatMinutes(usedMinutes)}</div>
          <div className={styles.metricNote}>Processed video this month</div>
        </article>
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Minutes remaining</div>
          <div className={styles.metricValue}>{remainingMinutes === null ? "—" : formatMinutes(remainingMinutes)}</div>
          <div className={styles.metricNote}>{monthlyLimit === null ? "Allowance not configured" : `${formatMinutes(monthlyLimit)} min monthly allowance`}</div>
        </article>
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Analyses</div>
          <div className={styles.metricValue}>{analysesCountResult.count ?? 0}</div>
          <div className={styles.metricNote}>Saved in this workspace</div>
        </article>
        <article className={styles.metricCard}>
          <div className={styles.metricLabel}>Processing now</div>
          <div className={styles.metricValue}>{activeResult.count ?? 0}</div>
          <div className={styles.metricNote}>Queued or running jobs</div>
        </article>
      </section>

      <div className={styles.contentGrid}>
        <div className={styles.stack}>
          <section className={styles.panel} aria-labelledby="recent-analyses-title">
            <div className={styles.panelHeader}>
              <div>
                <h2 id="recent-analyses-title">Recent analyses</h2>
                <p>Latest game and clip processing activity for this workspace. Open any run to follow processing or review its results.</p>
              </div>
              {canManageAnalysis ? <Link className={styles.secondaryButton} href="/analysis/new">New analysis</Link> : null}
            </div>

            {recentRuns.length ? (
              <div className={styles.analysisList}>
                {recentRuns.map((run) => {
                  const game = Array.isArray(run.games) ? run.games[0] : run.games;
                  const title = game?.title || sourceLabel(run.video_source);
                  const duration = formatDuration(run.start_time_sec, run.end_time_sec);
                  return (
                    <Link href={`/analysis/${run.id}`} key={run.id} style={{ color: "inherit", textDecoration: "none" }}>
                      <article className={styles.analysisRow}>
                        <div>
                          <p className={styles.analysisTitle}>{title}</p>
                          <div className={styles.analysisMeta}>
                            <span>{formatDate(game?.game_date || run.created_at)}</span>
                            {duration ? <span>{duration}</span> : null}
                            <span>{sourceLabel(run.video_source)}</span>
                          </div>
                        </div>
                        <span className={statusClass(run.status)}>{run.status}</span>
                      </article>
                    </Link>
                  );
                })}
              </div>
            ) : (
              <div className={styles.emptyState}>
                <strong>No analyses yet</strong>
                <p>Your completed, queued and running analyses will appear here once you process your first basketball video.</p>
                {canManageAnalysis ? (
                  <Link className={styles.primaryButton} href="/analysis/new">Run your first analysis</Link>
                ) : (
                  <span className={styles.disabledButton} aria-disabled="true">Analysis unavailable</span>
                )}
              </div>
            )}
          </section>

          <section className={styles.panel} aria-labelledby="usage-title">
            <div className={styles.panelHeader}>
              <div>
                <h2 id="usage-title">Monthly usage</h2>
                <p>Usage is measured from processed video duration rather than the number of game records.</p>
              </div>
              <span className={styles.planBadge}>{planName} · {entitlement?.status ?? "free"}</span>
            </div>

            <div className={styles.usageNumbers}>
              <strong>{formatMinutes(usedMinutes)}</strong>
              <span>{monthlyLimit === null ? "analysis minutes used" : `of ${formatMinutes(monthlyLimit)} analysis minutes`}</span>
            </div>
            {monthlyLimit !== null ? (
              <>
                <div className={styles.progressTrack} aria-label={`${Math.round(usagePercent)}% of monthly analysis allowance used`}>
                  <div className={styles.progressBar} style={{ width: `${usagePercent}%` }} />
                </div>
                <div className={styles.usageMeta}>
                  <span>{formatMinutes(remainingMinutes ?? 0)} min remaining</span>
                  <span>Resets with the monthly usage period</span>
                </div>
              </>
            ) : null}

            {usageLevel === "warning" ? (
              <div className={`${styles.usageNotice} ${styles.usageNoticeWarning}`} role="status">
                <strong>{Math.round(usagePercent)}% of this month&apos;s allowance used.</strong>
                <span>We warn at {USAGE_WARNING_PERCENT}% so you can prioritise the next games or clips. CourtFlux will not add automatic overage charges during private beta.</span>
              </div>
            ) : null}
            {usageLevel === "exhausted" ? (
              <div className={`${styles.usageNotice} ${styles.usageNoticeExhausted}`} role="status">
                <strong>Monthly analysis allowance reached.</strong>
                <span>New analyses are paused. A trusted CourtFlux admin can grant additional beta minutes; there is no automatic overage billing.</span>
              </div>
            ) : null}

            <div className={styles.betaNote}>
              Billing is not active during the private beta. {baseMonthlyLimit !== null ? `${planName} includes ${formatMinutes(baseMonthlyLimit)} analysis minutes per month.` : "Your base allowance is managed through the workspace entitlement."}
              {grantedMinutes > 0 ? ` This workspace currently has ${formatMinutes(grantedMinutes)} additional beta minutes.` : ""}
            </div>
            <div className={styles.planActions}>
              <Link className={styles.secondaryButton} href="/#pricing">Compare plans</Link>
              <span>No card required and no overage charges during private beta.</span>
            </div>
          </section>
        </div>

        <aside className={styles.stack} aria-label="Workspace details">
          <section className={styles.panel}>
            <div className={styles.eyebrow}>Workspace</div>
            <div className={styles.workspaceName}>{workspace?.name ?? "CourtFlux workspace"}</div>
            <div className={styles.detailList}>
              <div className={styles.detailRow}><span>Your role</span><strong>{role}</strong></div>
              <div className={styles.detailRow}><span>Plan</span><strong>{planName}</strong></div>
              <div className={styles.detailRow}><span>Monthly allowance</span><strong>{monthlyLimit === null ? "—" : `${formatMinutes(monthlyLimit)} min`}</strong></div>
              {grantedMinutes > 0 ? <div className={styles.detailRow}><span>Beta top-up</span><strong>+{formatMinutes(grantedMinutes)} min</strong></div> : null}
              <div className={styles.detailRow}><span>Teams</span><strong>{teamsResult.count ?? teams.length}</strong></div>
              <div className={styles.detailRow}><span>Account</span><strong>{user.email ?? "Signed in"}</strong></div>
              {entitlement?.current_period_end ? (
                <div className={styles.detailRow}><span>Current period</span><strong>Until {formatDate(entitlement.current_period_end)}</strong></div>
              ) : null}
            </div>

            {teams.length ? (
              <div className={styles.teamList} aria-label="Workspace teams">
                {teams.map((team) => <span className={styles.teamChip} key={team.id}>{team.short_name || team.name}</span>)}
              </div>
            ) : (
              <div className={styles.betaNote}>
                No team is configured yet. That does not block analysis: entering the two team names in New Analysis will create the match context automatically.
              </div>
            )}
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <h2>Quick actions</h2>
                <p>Customer workflows first; engineering diagnostics remain separate.</p>
              </div>
            </div>
            <div className={styles.quickActions}>
              {canManageAnalysis ? <Link href="/analysis/new">Start new analysis <span>→</span></Link> : <div className={styles.quickActionDisabled}>Analysis unavailable <span>{analysisAllowed ? "View-only role" : "Allowance used"}</span></div>}
              <Link href="/basketball-video-analysis">Basketball video analysis <span>→</span></Link>
              <Link href="/#pricing">Plans and analysis minutes <span>→</span></Link>
              <Link href="/lab">Engineering Beta Lab <span>→</span></Link>
            </div>
          </section>
        </aside>
      </div>
    </main>
  );
}
