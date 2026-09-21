import Link from "next/link";
import styles from "./foundation-results-preview.module.css";

type FoundationResultsPreviewProps = {
  runId: string;
  p3FoundationEnabled: boolean;
  p4SpatialFoundationEnabled: boolean;
};

const p3PreviewRows = [
  ["Turnover candidates", "—"],
  ["Shots needing review", "—"],
  ["Loose-ball sequences", "—"],
  ["Dunks", "—"],
  ["Alley-oop candidates", "—"],
] as const;

const p4PreviewRows = [
  ["Transition", "— possessions"],
  ["Pick-and-roll", "— actions"],
  ["Paint pressure", "— sequences"],
  ["Screens / cuts", "— examples"],
] as const;

export function FoundationResultsPreview({
  runId,
  p3FoundationEnabled,
  p4SpatialFoundationEnabled,
}: FoundationResultsPreviewProps) {
  return (
    <section className={styles.shell} aria-labelledby="future-results-title">
      <div className={styles.header}>
        <div>
          <div className={styles.eyebrow}>Prepared result layers</div>
          <h2 id="future-results-title">P3 review and P4 tactical results already have a home on this page.</h2>
          <p>
            These panels reserve the customer-facing result structure while the foundations are validated. Dashes are
            intentional placeholders, not zeroes and not inferred statistics from this run.
          </p>
        </div>
        <span className={styles.foundationPill}>Foundation preview</span>
      </div>

      <div className={styles.grid}>
        <article className={styles.layerCard}>
          <div className={styles.cardHeader}>
            <div>
              <div className={styles.phaseLine}>
                <span className={styles.phase}>P3</span>
                <span className={styles.state}>Foundation integrated</span>
              </div>
              <h3>Review & evidence</h3>
              <p>
                The expected P3 product opens a completed game into the moments that deserve attention, with confidence,
                provenance and physical evidence kept attached to every review item.
              </p>
            </div>
          </div>

          <div className={styles.previewPanel} aria-label="Expected P3 result layout">
            <div className={styles.previewTitle}>
              <div>
                <span>Expected result</span>
                <strong>Things to Review</strong>
              </div>
              <span className={styles.previewBadge}>Not populated yet</span>
            </div>
            <div className={styles.previewRows}>
              {p3PreviewRows.map(([label, value]) => (
                <div className={styles.previewRow} key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.capabilityBlock}>
            <strong>Prepared for</strong>
            <div className={styles.chips}>
              <span>Evidence clips</span>
              <span>Possession timeline</span>
              <span>Support / confidence</span>
              <span>Team & game-player splits</span>
              <span>Confirm / correct</span>
              <span>Save / tag / note</span>
            </div>
          </div>

          <div className={styles.boundary}>
            The foundation is structurally complete, but P2-dependent possession truth, final rates, ranking and basketball
            insights stay gated until the validated P2 baseline is ready.
          </div>

          <div className={styles.actions}>
            {p3FoundationEnabled ? (
              <Link href={`/analysis/${encodeURIComponent(runId)}/review`}>Open internal P3 review foundation</Link>
            ) : (
              <span>Internal P3 surface disabled</span>
            )}
          </div>
        </article>

        <article className={styles.layerCard}>
          <div className={styles.cardHeader}>
            <div>
              <div className={styles.phaseLine}>
                <span className={styles.phase}>P4</span>
                <span className={styles.state}>Spatial foundation complete</span>
              </div>
              <h3>Tactical review</h3>
              <p>
                The eventual P4 result explains how a team created advantages, then lets a coach open every supporting
                example with video and court evidence instead of hunting through the game manually.
              </p>
            </div>
          </div>

          <div className={styles.previewPanel} aria-label="Expected P4 tactical result layout">
            <div className={styles.previewTitle}>
              <div>
                <span>Expected result</span>
                <strong>How did we create advantage?</strong>
              </div>
              <span className={styles.previewBadge}>Tactics gated</span>
            </div>
            <div className={styles.previewRows}>
              {p4PreviewRows.map(([label, value]) => (
                <div className={styles.previewRow} key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.capabilityBlock}>
            <strong>Spatial foundation already supports</strong>
            <div className={styles.chips}>
              <span>FIBA court coordinates</span>
              <span>Geometry quality</span>
              <span>Spatial eligibility</span>
              <span>Smoothed trajectories</span>
              <span>Kinematics</span>
              <span>Relationships</span>
              <span>Team shape</span>
              <span>Motion primitives</span>
            </div>
          </div>

          <div className={styles.boundary}>
            The current P4 foundation deliberately does not infer possession, ball handler, offense/defense roles,
            matchups, screens, cuts, drives, pick-and-roll or tactical conclusions. Those arrive only after validated
            P2/P3 basketball context is joined to the spatial base.
          </div>

          <div className={styles.actions}>
            {p4SpatialFoundationEnabled ? (
              <Link href={`/analysis/${encodeURIComponent(runId)}/spatial`}>Open internal spatial diagnostics</Link>
            ) : (
              <span>Internal P4 spatial viewer disabled</span>
            )}
          </div>
        </article>
      </div>

      <div className={styles.footerNote}>
        <strong>Result-page rule:</strong> current evidence is shown as current evidence; expected P3/P4 outputs stay visibly
        marked as future layers until their derivations exist for the run. CourtFlux should never fill these slots with
        plausible-looking sample numbers.
      </div>
    </section>
  );
}
