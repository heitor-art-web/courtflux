import type { Metadata } from "next";
import Link from "next/link";
import { LegalPage, legalStyles as styles } from "../legal-page";

export const metadata: Metadata = {
  title: "Privacy Policy",
  description: "CourtFlux private-beta privacy policy covering accounts, basketball video, analysis data and retention.",
  alternates: { canonical: "/privacy" },
};

export default function PrivacyPage() {
  return (
    <LegalPage
      eyebrow="Privacy policy · Private beta"
      title="CourtFlux privacy policy"
      intro="This private-beta policy explains the data CourtFlux currently needs to operate the product and how uploaded basketball footage is handled. It is a working policy and will be reviewed before wider commercial launch."
      notice={<><strong>Key retention rule:</strong> uploaded source videos are scheduled for automatic deletion 60 days after upload. Derived analysis results and statistics remain in the workspace after the source video is removed.</>}
    >
      <section className={styles.section}>
        <h2>Who operates the service</h2>
        <p>The service is currently operated under the name <strong>CourtFlux</strong>. A formal legal-entity identity and dedicated privacy contact are launch-readiness items that have not yet been published.</p>
      </section>

      <section className={styles.section}>
        <h2>Data CourtFlux processes</h2>
        <ul>
          <li>Account information such as email address, profile name and authentication identifiers.</li>
          <li>Workspace information such as membership role, plan, teams and usage allowance.</li>
          <li>Basketball footage submitted directly or referenced through supported external sources.</li>
          <li>Game context supplied by users, including team names, dates, competition, kits and optional roster information.</li>
          <li>Derived analysis information such as tracking summaries, diagnostics, statistics, run status and processing logs.</li>
          <li>Technical information needed to secure and operate the service, including timestamps and service logs.</li>
        </ul>
      </section>

      <section className={styles.section}>
        <h2>How the data is used</h2>
        <p>CourtFlux uses this information to authenticate users, operate private workspaces, process requested basketball analyses, show results, measure analysis-minute usage, diagnose failures and improve the beta product.</p>
        <p>CourtFlux does not present uploaded customer footage as public content.</p>
      </section>

      <section className={styles.section}>
        <h2>Storage and service providers</h2>
        <p>The current web and data architecture uses service providers including Supabase for authentication, database and private storage, and Vercel for the customer-facing web application. Analysis workers process video separately from the web interface. Providers may process data as needed to deliver their infrastructure services.</p>
      </section>

      <section className={styles.section}>
        <h2>Video retention and analysis retention</h2>
        <p>Directly uploaded source videos are scheduled for deletion 60 days after upload. CourtFlux retains the derived analysis record and statistics so users can keep historical evidence after the original file is gone.</p>
        <p>Read the simpler <Link href="/video-data">Video & Data Policy</Link> for the product-level explanation.</p>
      </section>

      <section className={styles.section}>
        <h2>Children and youth basketball</h2>
        <p>CourtFlux is designed for organisations that may work with youth basketball. The organisation or user submitting footage is responsible for having the appropriate authority, notices and permissions to record, upload and analyse people appearing in that footage.</p>
      </section>

      <section className={styles.section}>
        <h2>Access, correction and deletion requests</h2>
        <p>A dedicated privacy email has not yet been published. During private beta, users should use the same communication channel through which they received CourtFlux access for account or data requests. A formal privacy contact and self-service deletion controls remain pre-launch work.</p>
      </section>
    </LegalPage>
  );
}
