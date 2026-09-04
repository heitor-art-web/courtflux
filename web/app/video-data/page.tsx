import type { Metadata } from "next";
import { LegalPage, legalStyles as styles } from "../legal-page";

export const metadata: Metadata = {
  title: "Video & Data Policy",
  description: "How CourtFlux stores, processes and removes uploaded basketball video while retaining analysis results.",
  alternates: { canonical: "/video-data" },
};

export default function VideoDataPage() {
  return (
    <LegalPage
      eyebrow="Video & data"
      title="Your uploaded video has a 60-day lifetime. Your analysis history does not disappear with it."
      intro="CourtFlux separates source footage from the structured evidence created from that footage. This keeps coaching results useful without storing source video indefinitely."
      notice={<><strong>Automatic source-video deletion:</strong> uploaded videos are scheduled for deletion 60 days after upload. Derived analysis summaries, metrics and workspace records are retained after the source file is deleted.</>}
    >
      <section className={styles.section}>
        <h2>What happens after upload</h2>
        <div className={styles.retention}>
          <div><span>Day 0</span><strong>Private upload and analysis</strong></div>
          <div><span>Days 1–60</span><strong>Source can be used for review and re-opening the analysis</strong></div>
          <div><span>After day 60</span><strong>Source video removed; analysis record stays</strong></div>
        </div>
        <p>The 60-day period is calculated from the upload time, not from the date of the basketball game.</p>
      </section>

      <section className={styles.section}>
        <h2>What remains after video deletion</h2>
        <ul>
          <li>The analysis run record and processing status.</li>
          <li>Derived tracking summaries and diagnostic statistics produced by the completed run.</li>
          <li>Game context such as team names, date and competition when supplied.</li>
          <li>Usage records needed to show analysis-minute consumption.</li>
        </ul>
        <p>The deleted source file is no longer available for playback in CourtFlux after retention expires.</p>
      </section>

      <section className={styles.section}>
        <h2>Uploaded footage and YouTube sources are different</h2>
        <p>For direct uploads, CourtFlux controls the private stored copy and applies the 60-day deletion policy. When a user submits a YouTube URL, the original source remains on YouTube under that platform&apos;s and uploader&apos;s control. CourtFlux may create temporary processing copies on worker infrastructure to perform analysis.</p>
      </section>

      <section className={styles.section}>
        <h2>Permission to upload</h2>
        <p>Users must have the right or authority to upload and analyse the footage they submit. This is particularly important for clubs, schools and academies working with children or other protected participants. The New Analysis flow asks the user to confirm this before an analysis can start.</p>
      </section>
    </LegalPage>
  );
}
