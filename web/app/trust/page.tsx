import type { Metadata } from "next";
import Link from "next/link";
import { LegalPage, legalStyles as styles } from "../legal-page";

export const metadata: Metadata = {
  title: "Trust & Security | CourtFlux",
  description: "How CourtFlux handles basketball video, workspace access, retention and beta AI limitations.",
  alternates: { canonical: "/trust" },
};

export default function TrustPage() {
  return (
    <LegalPage
      eyebrow="Trust at CourtFlux"
      title="Private basketball video should stay private — and the limits of the AI should stay visible."
      intro="CourtFlux is being built for real coaching workflows, including academies and youth basketball. This page explains the controls that exist today rather than claiming certifications or guarantees we do not yet have."
      notice={<><strong>60-day source-video retention.</strong> Videos uploaded to CourtFlux are automatically scheduled for deletion 60 days after upload. The derived analysis results and statistics remain available after the source video is removed.</>}
    >
      <section className={styles.section}>
        <h2>What is protected today</h2>
        <ul>
          <li>Uploaded analysis videos are stored in a private Supabase Storage bucket rather than a public media library.</li>
          <li>Analysis pages require authentication and are scoped to the user&apos;s active CourtFlux workspace.</li>
          <li>Private video playback uses short-lived signed URLs rather than permanent public links.</li>
          <li>Customer analysis routes are marked noindex so authenticated workspace content is not intended for search indexing.</li>
          <li>CourtFlux does not automatically publish uploaded footage or analysis results.</li>
        </ul>
      </section>

      <section className={styles.section}>
        <h2>Video retention</h2>
        <div className={styles.retention}>
          <div><span>Source video</span><strong>Deleted after 60 days</strong></div>
          <div><span>Analysis results</span><strong>Retained after video deletion</strong></div>
          <div><span>Reason</span><strong>Keep review history without storing footage indefinitely</strong></div>
        </div>
        <p>Deletion applies to uploaded CourtFlux source videos. If a user analyses a YouTube URL, the original video remains governed by YouTube and the uploader; CourtFlux may create temporary processing copies on its worker infrastructure.</p>
      </section>

      <section className={styles.section}>
        <h2>What CourtFlux does not claim</h2>
        <p>During private beta, CourtFlux does not claim SOC 2 certification, ISO certification, end-to-end encryption, legal compliance certification, or perfect computer-vision accuracy. Security and compliance controls will mature with the product and will be documented when they are actually in place.</p>
        <p>See <Link href="/ai-limitations">AI limitations</Link> for the current evidence boundary.</p>
      </section>

      <section className={styles.section}>
        <h2>Youth basketball and safeguarding</h2>
        <p>Academies, clubs and other organisations uploading footage that includes minors are responsible for having the authority, notices and permissions required for that recording and its analysis. CourtFlux should not be used to bypass a club&apos;s safeguarding or privacy obligations.</p>
      </section>
    </LegalPage>
  );
}
