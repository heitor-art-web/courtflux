import type { Metadata } from "next";
import Link from "next/link";
import { LegalPage, legalStyles as styles } from "../legal-page";

export const metadata: Metadata = {
  title: "Terms of Use",
  description: "Working private-beta terms for using CourtFlux basketball video analysis.",
  alternates: { canonical: "/terms" },
};

export default function TermsPage() {
  return (
    <LegalPage
      eyebrow="Terms of use · Private beta"
      title="CourtFlux private-beta terms"
      intro="These are working product terms for the private beta, not a substitute for the formal commercial terms that will be reviewed before public launch."
      notice={<><strong>Using CourtFlux means submitting only footage you are authorised to analyse.</strong> Uploaded source video is retained for up to 60 days and then removed automatically; derived analysis results and statistics remain available.</>}
    >
      <section className={styles.section}>
        <h2>1. Private-beta service</h2>
        <p>CourtFlux is under active development. Features can change, processing can fail, capacity can be limited and some product areas remain experimental or unavailable. Beta access does not guarantee uninterrupted service or a specific future feature.</p>
      </section>

      <section className={styles.section}>
        <h2>2. Your footage and permissions</h2>
        <p>You keep responsibility for the footage and match information you submit. You must have the rights, permission or organisational authority needed to upload, process and review that footage with CourtFlux.</p>
        <p>Do not use CourtFlux to upload footage unlawfully or to bypass safeguarding, competition, venue, club, school or privacy requirements.</p>
      </section>

      <section className={styles.section}>
        <h2>3. Youth basketball</h2>
        <p>If footage contains minors, the club, academy, school, coach or other organisation submitting it is responsible for ensuring the recording and analysis are permitted under the rules that apply to them.</p>
      </section>

      <section className={styles.section}>
        <h2>4. Source-video retention</h2>
        <p>Uploaded source video is scheduled for deletion 60 days after upload. CourtFlux retains derived analysis results, statistics, game context and usage records after that deletion so the historical analysis remains useful.</p>
        <p>See the <Link href="/video-data">Video & Data Policy</Link> for more detail.</p>
      </section>

      <section className={styles.section}>
        <h2>5. AI output and basketball decisions</h2>
        <p>CourtFlux output is probabilistic and can be incomplete or wrong. Tracking quality depends on the video and the current state of the models. Coaches and analysts remain responsible for basketball interpretation and decisions.</p>
        <p>See <Link href="/ai-limitations">AI limitations</Link> for the current product boundary.</p>
      </section>

      <section className={styles.section}>
        <h2>6. Acceptable use</h2>
        <p>Users must not attempt to gain unauthorised access to another workspace, interfere with the service, upload malicious files, abuse worker capacity, or use CourtFlux for unlawful surveillance or other unlawful purposes.</p>
      </section>

      <section className={styles.section}>
        <h2>7. Billing during beta</h2>
        <p>Billing is not currently active in the private beta. Analysis-minute allowances are product limits used to manage processing capacity. CourtFlux does not automatically charge overages during the current beta.</p>
      </section>

      <section className={styles.section}>
        <h2>8. Pre-launch legal review</h2>
        <p>The registered legal entity, business address, dedicated privacy/support contact and final commercial terms remain launch-readiness items. CourtFlux will update these terms before broader paid availability.</p>
      </section>
    </LegalPage>
  );
}
