"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import {
  ONBOARDING_ROLE_LABELS,
  type OnboardingRole,
} from "@/lib/onboarding";
import styles from "./page.module.css";

type Props = {
  displayName: string;
  defaultWorkspaceName: string;
  initialRole: OnboardingRole | null;
};

type SetupResult = {
  ok?: boolean;
  error?: string;
  workspace?: { id: string; name: string };
  team?: { id: string; name: string };
};

const roleCards: Array<{
  code: OnboardingRole;
  title: string;
  description: string;
}> = [
  {
    code: "coach",
    title: ONBOARDING_ROLE_LABELS.coach,
    description: "I coach a team and want faster review of game film.",
  },
  {
    code: "analyst",
    title: ONBOARDING_ROLE_LABELS.analyst,
    description: "I work with video, performance data or opposition analysis.",
  },
  {
    code: "team_staff",
    title: ONBOARDING_ROLE_LABELS.team_staff,
    description: "I support a club or team workflow with coaching staff.",
  },
  {
    code: "academy_staff",
    title: ONBOARDING_ROLE_LABELS.academy_staff,
    description: "I work across player development, academy or programme teams.",
  },
  {
    code: "other",
    title: ONBOARDING_ROLE_LABELS.other,
    description: "I have another basketball role and want to analyse game film.",
  },
];

export function OnboardingFlow({ displayName, defaultWorkspaceName, initialRole }: Props) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [role, setRole] = useState<OnboardingRole | null>(initialRole);
  const [teamName, setTeamName] = useState("");
  const [workspaceName, setWorkspaceName] = useState(defaultWorkspaceName);
  const [savedTeam, setSavedTeam] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function finishSetup() {
    if (!role || teamName.trim().length < 2) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          basketball_role: role,
          team_name: teamName,
          workspace_name: workspaceName,
        }),
      });
      const payload = (await response.json()) as SetupResult;
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error || "Could not finish workspace setup");
      }
      setSavedTeam(payload.team?.name || teamName.trim());
      setStep(3);
    } catch (setupError) {
      setError(setupError instanceof Error ? setupError.message : String(setupError));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}>
        <Link href="/" className={styles.brand} aria-label="CourtFlux home">
          <Image
            src="/brand/horizontal_logo_safe_crop.png"
            alt="CourtFlux — Basketball Intelligence"
            width={660}
            height={185}
            priority
          />
        </Link>
        <span className={styles.betaBadge}>Private beta setup</span>
      </header>

      <div className={styles.shell}>
        <section className={styles.progress} aria-label="Onboarding progress">
          {[
            [1, "Role"],
            [2, "Team"],
            [3, "First analysis"],
          ].map(([number, label]) => {
            const numeric = number as number;
            const active = step === numeric;
            const complete = step > numeric;
            return (
              <div className={`${styles.progressItem} ${active ? styles.progressActive : ""} ${complete ? styles.progressDone : ""}`} key={label}>
                <span>{complete ? "✓" : number}</span>
                <strong>{label}</strong>
              </div>
            );
          })}
        </section>

        {step === 1 ? (
          <section className={styles.card} aria-labelledby="role-title">
            <div className={styles.eyebrow}>Welcome to CourtFlux</div>
            <h1 id="role-title">Hi {displayName.split(/\s+/)[0]}. What best describes your role?</h1>
            <p className={styles.lead}>
              This helps CourtFlux shape the workspace around the way you review basketball. It does not change your account permissions or analysis quality.
            </p>

            <div className={styles.roleGrid} role="radiogroup" aria-label="Basketball role">
              {roleCards.map((item) => (
                <button
                  key={item.code}
                  type="button"
                  role="radio"
                  aria-checked={role === item.code}
                  className={`${styles.roleCard} ${role === item.code ? styles.roleSelected : ""}`}
                  onClick={() => setRole(item.code)}
                >
                  <span className={styles.roleCheck}>{role === item.code ? "✓" : ""}</span>
                  <strong>{item.title}</strong>
                  <small>{item.description}</small>
                </button>
              ))}
            </div>

            <div className={styles.actions}>
              <button className={styles.primaryButton} type="button" disabled={!role} onClick={() => setStep(2)}>
                Continue to team setup
              </button>
            </div>
          </section>
        ) : null}

        {step === 2 ? (
          <section className={styles.card} aria-labelledby="team-title">
            <button className={styles.backButton} type="button" onClick={() => setStep(1)}>← Back</button>
            <div className={styles.eyebrow}>Your first team</div>
            <h1 id="team-title">Set up the team you work with most.</h1>
            <p className={styles.lead}>
              Just the team name is needed now. Rosters, jersey numbers and detailed team management can be added later; they never block your first analysis.
            </p>

            <div className={styles.formGrid}>
              <div className={styles.field}>
                <label htmlFor="team-name">Team name *</label>
                <input
                  id="team-name"
                  value={teamName}
                  onChange={(event) => setTeamName(event.target.value)}
                  placeholder="e.g. Porto U18"
                  autoFocus
                  maxLength={80}
                />
                <small>Your opponent is added when you start an analysis.</small>
              </div>
              <div className={styles.field}>
                <label htmlFor="workspace-name">Workspace name</label>
                <input
                  id="workspace-name"
                  value={workspaceName}
                  onChange={(event) => setWorkspaceName(event.target.value)}
                  placeholder="CourtFlux workspace"
                  maxLength={80}
                />
                <small>This is the private area where your analyses and teams are saved.</small>
              </div>
            </div>

            <div className={styles.infoBox}>
              <strong>You can start before the roster is perfect.</strong>
              <span>CourtFlux only needs home and away team names for the first analysis. Player identity and roster detail remain optional.</span>
            </div>

            {error ? <div className={styles.errorBox} role="alert">{error}</div> : null}

            <div className={styles.actions}>
              <button
                className={styles.primaryButton}
                type="button"
                disabled={busy || teamName.trim().length < 2 || workspaceName.trim().length < 2}
                onClick={() => void finishSetup()}
                aria-busy={busy}
              >
                {busy ? "Saving workspace…" : "Save team and continue"}
              </button>
            </div>
          </section>
        ) : null}

        {step === 3 ? (
          <section className={`${styles.card} ${styles.readyCard}`} aria-labelledby="ready-title">
            <span className={styles.readyMark}>✓</span>
            <div className={styles.eyebrow}>Workspace ready</div>
            <h1 id="ready-title">You&apos;re ready for your first CourtFlux analysis.</h1>
            <p className={styles.lead}>
              {savedTeam ? `${savedTeam} is saved in your workspace. ` : "Your first team is saved. "}
              Add a game video, choose the part you want processed and CourtFlux will show the analysis-minute estimate before anything starts.
            </p>

            <div className={styles.readyGrid}>
              <article><span>Beta allowance</span><strong>60 min / month</strong><small>No card or automatic overage charges.</small></article>
              <article><span>Video retention</span><strong>60 days</strong><small>Uploaded source video is deleted; derived stats remain.</small></article>
              <article><span>Match setup</span><strong>2 team names</strong><small>Roster and jersey data are optional.</small></article>
            </div>

            <div className={styles.actions}>
              <Link className={styles.primaryButton} href="/analysis/new">Start first analysis</Link>
              <Link className={styles.secondaryButton} href="/account">Go to dashboard</Link>
            </div>

            <p className={styles.policyNote}>
              Before submitting footage, CourtFlux asks you to confirm you are authorised to analyse it. Read the <Link href="/video-data">Video & Data Policy</Link> or <Link href="/trust">Trust page</Link> at any time.
            </p>
          </section>
        ) : null}
      </div>
    </main>
  );
}
