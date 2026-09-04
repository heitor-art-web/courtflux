import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import styles from "./home.module.css";
import { getSiteUrl } from "@/lib/site";

const siteUrl = getSiteUrl();

export const metadata: Metadata = {
  title: "Basketball Video Analysis for Coaches, Teams & Academies",
  description:
    "Turn basketball game film into structured player, ball and spatial intelligence. CourtFlux helps coaches, teams and academies reach useful evidence faster.",
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    url: "/",
    siteName: "CourtFlux",
    title: "CourtFlux — Basketball Video Analysis & Intelligence",
    description:
      "Turn basketball game film into structured player, ball and spatial intelligence for coaches, teams and academies.",
    images: [
      {
        url: "/brand/CourtFlux_Full_Analyst_Workspace.png",
        width: 1672,
        height: 941,
        alt: "CourtFlux basketball video analysis workspace",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "CourtFlux — Basketball Video Analysis & Intelligence",
    description:
      "Turn basketball game film into structured player, ball and spatial intelligence.",
    images: ["/brand/CourtFlux_Full_Analyst_Workspace.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

const capabilities = [
  ["Computer Vision", "Player Tracking", "Detect and persist players across game footage.", "Beta"],
  ["Core Priority", "Ball Intelligence", "Detect, associate and maintain the ball state across time.", "Beta"],
  ["Classification", "Team Identification", "Separate teams consistently from visual appearance.", "Beta"],
  ["Spatial Layer", "Court Mapping", "Translate image positions into meaningful court coordinates.", "In development"],
  ["Game Structure", "Possession Analysis", "Segment continuous footage into basketball possessions.", "In development"],
  ["Coaching Layer", "Tactical Intelligence", "Explain patterns and moments in coaching language.", "Vision"],
] as const;

const faq = [
  [
    "What kind of video can CourtFlux analyse?",
    "CourtFlux is being developed around basketball game footage where the court and players are sufficiently visible. Camera position, resolution, occlusion and motion blur can affect analysis quality.",
  ],
  [
    "Do I need specialist cameras?",
    "No proprietary capture system is required by the product direction. CourtFlux is designed around normal basketball game film.",
  ],
  [
    "How do analysis minutes work?",
    "Usage is based on processed video duration rather than the number of uploads. A 3-minute clip uses about 3 analysis minutes and a 60-minute recording uses about 60.",
  ],
  [
    "What counts as a full game?",
    "For pricing guidance, a full game means up to 60 minutes of uploaded video. A game can still be stored as a matchup or session in the product; usage is based on the actual video duration processed.",
  ],
  [
    "What is available in the beta today?",
    "The current benchmark focuses on player detection and tracking, team classification, ball detection, temporal ball tracking and processing diagnostics. Court mapping, possessions, events and tactical intelligence are being developed in stages.",
  ],
  [
    "Does AI replace the coach or analyst?",
    "No. CourtFlux is intended to surface evidence and reduce manual video work while keeping basketball judgement with the coach or analyst.",
  ],
] as const;

function CapabilityCard({ item }: { item: (typeof capabilities)[number] }) {
  return (
    <article className={styles.card}>
      <div className={styles.kicker}>{item[0]}</div>
      <h3>{item[1]}</h3>
      <p>{item[2]}</p>
      <span className={styles.badge}>{item[3]}</span>
    </article>
  );
}

export default function HomePage() {
  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "Organization",
        "@id": `${siteUrl}/#organization`,
        name: "CourtFlux",
        url: siteUrl,
        logo: `${siteUrl}/brand/primary_logo_safe_crop.png`,
        description: "Basketball video analysis and intelligence for coaches, teams and academies.",
      },
      {
        "@type": "SoftwareApplication",
        "@id": `${siteUrl}/#software`,
        name: "CourtFlux",
        url: siteUrl,
        applicationCategory: "SportsApplication",
        operatingSystem: "Web",
        description:
          "CourtFlux turns basketball game film into structured player, ball, spatial and coaching intelligence.",
        publisher: { "@id": `${siteUrl}/#organization` },
      },
    ],
  };

  return (
    <main className={styles.home}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />

      <header className={styles.siteHeader}>
        <Link href="/" className={styles.logoLink} aria-label="CourtFlux home">
          <Image
            className={styles.logo}
            src="/brand/horizontal_logo_safe_crop.png"
            alt="CourtFlux — Basketball Intelligence"
            width={660}
            height={185}
            priority
          />
        </Link>
        <nav className={styles.nav} aria-label="Primary navigation">
          <a href="#product">Product</a>
          <a href="#how">How it works</a>
          <a href="#pricing">Pricing</a>
          <Link href="/lab">Beta Lab</Link>
          <Link href="/login">Sign in</Link>
          <Link className={styles.navCta} href="/login?mode=signup">Start free beta</Link>
        </nav>
      </header>

      <div className={styles.shell}>
        <section className={styles.hero} aria-labelledby="hero-title">
          <div>
            <div className={styles.eyebrow}>Basketball intelligence from ordinary game film</div>
            <h1 id="hero-title">
              From game film to <span>decisions coaches can use.</span>
            </h1>
            <p className={styles.heroCopy}>
              Upload the footage you already have. CourtFlux follows players and the ball, builds spatial
              context, and turns movement into structured basketball intelligence.
            </p>
            <div className={styles.actions}>
              <Link className={styles.primaryButton} href="/login?mode=signup">Start free beta</Link>
              <a className={styles.secondaryButton} href="#how">See how it works</a>
            </div>
            <div className={styles.pills} aria-label="Product principles">
              <span className={styles.pill}>No proprietary camera required</span>
              <span className={styles.pill}>Coach-first AI</span>
              <span className={styles.pill}>Private beta</span>
            </div>
          </div>

          <div className={styles.visualFrame}>
            <Image
              src="/brand/CourtFlux_Full_Analyst_Workspace.png"
              alt="CourtFlux analyst workspace showing basketball video, tracked players, ball movement and analysis panels"
              width={1672}
              height={941}
              sizes="(max-width: 900px) 100vw, 58vw"
              priority
            />
          </div>
        </section>

        <section className={styles.signalGrid} aria-label="CourtFlux workflow">
          <div className={styles.signal}><strong>Use existing footage</strong><span>Start from normal basketball game video.</span></div>
          <div className={styles.signal}><strong>Track the game</strong><span>Players, teams and ball over time.</span></div>
          <div className={styles.signal}><strong>Build spatial context</strong><span>Translate pixels into court movement.</span></div>
          <div className={styles.signal}><strong>Reach useful evidence</strong><span>Surface moments coaches can review.</span></div>
        </section>

        <section id="how" className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>How it works</div>
            <h2 className={styles.sectionTitle}>One video. Three layers of <span>understanding.</span></h2>
            <p className={styles.sectionCopy}>
              CourtFlux bridges raw game footage and the spatial, temporal and tactical information hidden inside each possession.
            </p>
          </div>

          <div className={styles.split}>
            <div className={styles.visualFrame}>
              <Image
                src="/brand/CourtFlux_Video_Tracking_Tactical_Insight.png"
                alt="Basketball video progressing from tracking to tactical insight in CourtFlux"
                width={768}
                height={1024}
                sizes="(max-width: 900px) 100vw, 52vw"
              />
            </div>
            <div>
              <div className={styles.stack}>
                <article className={styles.step}><div className={styles.kicker}>01 · Video</div><h3>Use the game film you already have.</h3><p>Work from normal basketball footage rather than a proprietary capture system.</p></article>
                <article className={styles.step}><div className={styles.kicker}>02 · Tracking</div><h3>Computer vision follows the game.</h3><p>Players, teams and ball detections become persistent objects with temporal context.</p></article>
                <article className={styles.step}><div className={styles.kicker}>03 · Intelligence</div><h3>Movement becomes basketball information.</h3><p>Coordinates, possessions, events and tactical patterns become measurable and searchable as the product develops.</p></article>
              </div>
              <div className={styles.contextBox}>
                <strong>Match context is progressive, not a blocker.</strong>
                <p>Home and away teams are enough to start. Jersey numbers and player names can be added later when roster-level identity matters.</p>
              </div>
            </div>
          </div>
        </section>

        <section id="product" className={styles.section}>
          <div className={styles.splitReverse + " " + styles.split}>
            <div>
              <div className={styles.eyebrow}>Ball Intelligence</div>
              <h2 className={styles.sectionTitle}>Follow the object that defines the <span>possession.</span></h2>
              <p className={styles.sectionCopy}>
                Ball quality is a foundational dependency for possession, passing, shot, rebound and tactical analysis. CourtFlux combines direct detections with temporal continuity to maintain a more reliable ball state.
              </p>
              <ul className={styles.bulletList}>
                <li><strong>Direct detection</strong> — frame-level ball candidates</li>
                <li><strong>Temporal tracking</strong> — continuity through short gaps</li>
                <li><strong>Reacquisition</strong> — recovery when the ball becomes visible again</li>
              </ul>
            </div>
            <div className={styles.visualFrame}>
              <Image
                src="/brand/CourtFlux_Ball_Intelligence.png"
                alt="CourtFlux basketball ball detection and temporal tracking visualization"
                width={768}
                height={1024}
                sizes="(max-width: 900px) 100vw, 48vw"
              />
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.split}>
            <div className={styles.visualFrame}>
              <Image
                src="/brand/CourtFlux_Split_Screen_Court_Reconstruction.png"
                alt="Basketball video and reconstructed court positions shown side by side in CourtFlux"
                width={1672}
                height={941}
                sizes="(max-width: 900px) 100vw, 54vw"
              />
            </div>
            <div>
              <div className={styles.eyebrow}>Court Reconstruction</div>
              <h2 className={styles.sectionTitle}>From broadcast pixels to <span>spatial intelligence.</span></h2>
              <p className={styles.sectionCopy}>
                Map tracked objects from video coordinates onto a basketball court so coaches can reason about spacing, coverage, trajectories and team shape instead of isolated boxes.
              </p>
              <p className={styles.smallNote}>Court mapping is in development and is not represented as a finished beta capability.</p>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>Product capabilities</div>
            <h2 className={styles.sectionTitle}>Built around the <span>game itself.</span></h2>
            <p className={styles.sectionCopy}>
              CourtFlux is not a traditional box-score tracker. The platform is built around objects, movement, possessions and spatial context extracted from video.
            </p>
          </div>
          <div className={styles.cardGrid}>
            {capabilities.map((item) => <CapabilityCard key={item[1]} item={item} />)}
          </div>
        </section>

        <section id="pricing" className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>Pricing</div>
            <h2 className={styles.sectionTitle}>Pay for the video you <span>actually analyse.</span></h2>
            <p className={styles.sectionCopy}>
              Plans are structured around analysis minutes rather than a fixed number of games. Short clips consume only their duration, while full-game recordings use a larger share of the same monthly allowance.
            </p>
          </div>

          <div className={styles.pricingGrid}>
            <article className={styles.priceCard}>
              <div className={styles.kicker}>Private beta</div>
              <h3>Beta</h3>
              <div className={styles.price}>€0 <small>/ month</small></div>
              <p className={styles.priceDescription}>Explore the core tracking layer while CourtFlux is being benchmarked and refined.</p>
              <div className={styles.usage}><strong>60 analysis minutes / month</strong><span>About one full-game recording, or several shorter clips.</span></div>
              <ul className={styles.featureList}><li>Player tracking</li><li>Team classification</li><li>Ball intelligence</li><li>Basic analysis and diagnostics</li></ul>
              <Link className={styles.secondaryButton} href="/login?mode=signup">Start free beta</Link>
            </article>

            <article className={styles.priceCard}>
              <div className={styles.kicker}>Individual workflow</div>
              <h3>Coach</h3>
              <div className={styles.price}>€14.99 <small>/ month</small></div>
              <p className={styles.priceDescription}>For an individual coach or analyst who wants useful basketball moments without hours of manual review.</p>
              <div className={styles.usage}><strong>300 analysis minutes / month</strong><span>About five full-game recordings, or dozens of shorter clips.</span></div>
              <ul className={styles.featureList}><li>Everything in Beta</li><li>Possession and event primitives as they launch</li><li>Saved analyses and game summaries</li><li>One coach workspace</li></ul>
              <Link className={styles.secondaryButton} href="/login?mode=signup">Join the beta</Link>
            </article>

            <article className={`${styles.priceCard} ${styles.priceCardFeatured}`}>
              <span className={styles.priceTag}>Best for teams</span>
              <div className={styles.kicker}>Shared team workflow</div>
              <h3>Team</h3>
              <div className={styles.price}>€29.99 <small>/ month</small></div>
              <p className={styles.priceDescription}>For one squad and coaching staff building a repeatable season-long analysis workflow.</p>
              <div className={styles.usage}><strong>720 analysis minutes / month</strong><span>About twelve full-game recordings, with clips charged only by duration.</span></div>
              <ul className={styles.featureList}><li>Everything in Coach</li><li>Saved roster and player identity</li><li>Structured player and team analytics</li><li>Shared staff access and exports</li></ul>
              <Link className={styles.primaryButton} href="/login?mode=signup">Join the beta</Link>
            </article>

            <article className={styles.priceCard}>
              <div className={styles.kicker}>Multi-team development</div>
              <h3>Academy</h3>
              <div className={styles.price}>From €79 <small>/ month</small></div>
              <p className={styles.priceDescription}>For academies and clubs managing multiple squads, staff members and player development pathways.</p>
              <div className={styles.usage}><strong>2,400 analysis minutes / month</strong><span>About forty full-game recordings across the organisation.</span></div>
              <ul className={styles.featureList}><li>Up to 5 teams</li><li>Multiple rosters and staff permissions</li><li>Player development history and cross-team reporting</li><li>Tactical intelligence as it becomes available</li></ul>
              <Link className={styles.secondaryButton} href="/login?mode=signup">Talk to CourtFlux</Link>
            </article>
          </div>

          <div className={styles.usageExplainer}>
            <div>
              <strong>What counts as a game?</strong>
              <p>A game is a basketball session in your library, usually tied to one matchup and date. Usage is based on processed video duration, not the number of game records.</p>
            </div>
            <div>
              <div className={styles.usageScale}><span>Short clip · up to 5 min</span><span>Segment · 5–20 min</span><span>Full game · up to 60 min</span></div>
              <p>Footage over 60 minutes simply continues against the same analysis-minute allowance.</p>
            </div>
          </div>
          <p className={styles.pricingNote}>Indicative product pricing only. Billing is not active during private beta. Final allowances, feature availability and paid launch terms will be confirmed before charging customers.</p>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>Beta transparency</div>
            <h2 className={styles.sectionTitle}>Build the foundation before the <span>promise.</span></h2>
            <p className={styles.sectionCopy}>The public product story separates what is benchmarked from active development and the longer-term coaching vision.</p>
          </div>
          <div className={styles.statusGrid}>
            <article className={styles.statusCard}><div className={styles.kicker}>Available / benchmarked</div><ul><li><span className={styles.now}>●</span> Player detection and tracking</li><li><span className={styles.now}>●</span> Team classification</li><li><span className={styles.now}>●</span> Ball detection</li><li><span className={styles.now}>●</span> Temporal ball tracking</li><li><span className={styles.now}>●</span> Processing diagnostics</li></ul></article>
            <article className={styles.statusCard}><div className={styles.kicker}>In development</div><ul><li><span className={styles.doing}>●</span> Player identity</li><li><span className={styles.doing}>●</span> Court mapping</li><li><span className={styles.doing}>●</span> Possession segmentation</li><li><span className={styles.doing}>●</span> Basketball event detection</li></ul></article>
            <article className={styles.statusCard}><div className={styles.kicker}>Product vision</div><ul><li><span className={styles.later}>●</span> Automated tactical insights</li><li><span className={styles.later}>●</span> Team pattern analysis</li><li><span className={styles.later}>●</span> Player development intelligence</li><li><span className={styles.later}>●</span> High-performance workflows</li></ul></article>
          </div>

          <div className={styles.betaCta}>
            <div className={styles.eyebrow}>CourtFlux Beta Lab</div>
            <h3>The test environment stays a test environment.</h3>
            <p>Configure clips, tune detector thresholds, run the pipeline and inspect diagnostics without mixing engineering controls into the customer-facing marketing experience.</p>
            <div className={styles.actions}><Link className={styles.secondaryButton} href="/lab">Open Beta Lab</Link></div>
          </div>
        </section>

        <section className={styles.section} id="faq">
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>FAQ</div>
            <h2 className={styles.sectionTitle}>Questions about <span>CourtFlux.</span></h2>
          </div>
          <div className={styles.faqGrid}>
            {faq.map(([question, answer]) => (
              <details key={question}>
                <summary>{question}</summary>
                <p>{answer}</p>
              </details>
            ))}
          </div>
        </section>

        <section className={styles.finalCta}>
          <div className={styles.eyebrow}>CourtFlux</div>
          <h2 className={styles.sectionTitle}>See more in the same <span>game film.</span></h2>
          <p className={styles.sectionCopy}>Turn basketball video into a structured foundation for tracking, analysis and better coaching decisions.</p>
          <div className={styles.actions}><Link className={styles.primaryButton} href="/login?mode=signup">Start free beta</Link><a className={styles.secondaryButton} href="#pricing">See pricing</a></div>
        </section>

        <footer className={styles.footer}>
          <div className={styles.footerInner}>
            <div>
              <Image className={styles.footerLogo} src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux — Basketball Intelligence" width={660} height={185} />
              <div className={styles.footerMeta}>© 2026 CourtFlux · Basketball Intelligence</div>
            </div>
            <div className={styles.footerLinks}><a href="#product">Product</a><a href="#pricing">Pricing</a><a href="#faq">FAQ</a><Link href="/login">Sign in</Link><Link href="/lab">Beta Lab</Link></div>
          </div>
        </footer>
      </div>
    </main>
  );
}
