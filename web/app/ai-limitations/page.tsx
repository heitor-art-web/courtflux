import type { Metadata } from "next";
import { LegalPage, legalStyles as styles } from "../legal-page";

export const metadata: Metadata = {
  title: "AI Limitations | CourtFlux",
  description: "Current limitations of CourtFlux basketball tracking and beta analysis output.",
  alternates: { canonical: "/ai-limitations" },
};

export default function AiLimitationsPage() {
  return (
    <LegalPage
      eyebrow="AI limitations"
      title="CourtFlux evidence is useful only when its uncertainty stays visible."
      intro="Computer vision does not observe basketball perfectly. Camera angle, occlusion, motion blur, resolution, crowded paint areas and rapid ball movement can all reduce tracking quality."
      notice={<><strong>Beta evidence boundary:</strong> current customer results focus on player, team and ball tracking diagnostics. Possessions, passes, shots, rebounds, event attribution and tactical recommendations are not presented as reliable results until those layers are validated.</>}
    >
      <section className={styles.section}>
        <h2>Direct evidence and predicted continuity are different</h2>
        <p>A direct ball detection comes from an observed frame-level candidate selected by the tracking system. A predicted hold is a conservative temporal estimate used to bridge a short gap. CourtFlux keeps those categories separate in results so predicted continuity is not presented as a direct observation.</p>
      </section>

      <section className={styles.section}>
        <h2>Common failure conditions</h2>
        <ul>
          <li>The ball is hidden behind a player, referee or basket structure.</li>
          <li>The camera pans, zooms or cuts quickly.</li>
          <li>Video resolution, compression or lighting makes the ball difficult to distinguish.</li>
          <li>Players overlap heavily or leave and re-enter the frame.</li>
          <li>Similar kit colours make team classification harder.</li>
          <li>Jersey numbers are too small, blurred or occluded to read reliably.</li>
        </ul>
      </section>

      <section className={styles.section}>
        <h2>Tracking confidence is not basketball truth</h2>
        <p>Model confidence describes the system&apos;s certainty about a computer-vision output. It is not a guarantee that an event happened, that a player identity is correct, or that a coaching interpretation is valid.</p>
      </section>

      <section className={styles.section}>
        <h2>Coach and analyst judgement remains necessary</h2>
        <p>CourtFlux is designed to reduce manual video work and surface evidence, not to replace basketball judgement. Important decisions should be checked against the source footage and the user&apos;s own context.</p>
      </section>
    </LegalPage>
  );
}
