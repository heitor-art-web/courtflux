import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import styles from "./page.module.css";
import { getSiteUrl } from "@/lib/site";

const siteUrl = getSiteUrl();
const path = "/basketball-video-analysis";

export const metadata: Metadata = {
  title: "Basketball Video Analysis Software for Coaches",
  description:
    "Turn ordinary basketball game film into tracked player, team and ball evidence. CourtFlux helps coaches, teams and academies analyse video faster.",
  alternates: {
    canonical: path,
  },
  openGraph: {
    type: "website",
    url: path,
    siteName: "CourtFlux",
    title: "Basketball Video Analysis Software for Coaches | CourtFlux",
    description:
      "Use the basketball footage you already record to build player, team and ball evidence for faster video review.",
    images: [
      {
        url: "/brand/CourtFlux_Full_Analyst_Workspace.png",
        width: 1672,
        height: 941,
        alt: "CourtFlux basketball video analysis workspace with tracked players, ball movement and analysis panels",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Basketball Video Analysis Software for Coaches | CourtFlux",
    description:
      "Use the basketball footage you already record to build player, team and ball evidence for faster video review.",
    images: ["/brand/CourtFlux_Full_Analyst_Workspace.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

const faq = [
  [
    "What is basketball video analysis software?",
    "Basketball video analysis software helps coaches and analysts turn game footage into structured information they can review. CourtFlux is being built to extract player, team, ball and eventually spatial and event context directly from video.",
  ],
  [
    "Do I need a special camera system?",
    "CourtFlux is designed around normal basketball game film rather than a proprietary capture system. Analysis quality still depends on factors such as camera position, resolution, visibility, occlusion and motion blur.",
  ],
  [
    "What does CourtFlux analyse today?",
    "The current beta focuses on player detection and tracking, team classification, ball detection and temporal ball tracking. Court mapping, possession segmentation, basketball events, player attribution and tactical intelligence are being developed in stages.",
  ],
  [
    "Can I analyse a short basketball clip instead of a full game?",
    "Yes. CourtFlux uses analysis minutes based on processed video duration, so a short clip consumes only its own duration rather than being treated as a full game.",
  ],
  [
    "Is CourtFlux for individual coaches or whole organisations?",
    "The product direction supports individual coaches, single teams and multi-team academies. The underlying analysis quality is not intentionally reduced by plan; higher tiers are intended to add scale, history, collaboration and organisation workflows.",
  ],
] as const;

export default function BasketballVideoAnalysisPage() {
  const pageUrl = `${siteUrl}${path}`;
  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": `${pageUrl}#webpage`,
        url: pageUrl,
        name: "Basketball Video Analysis Software for Coaches",
        description:
          "CourtFlux turns ordinary basketball game film into tracked player, team and ball evidence for coaches, teams and academies.",
        isPartOf: { "@id": `${siteUrl}/#website` },
        about: { "@id": `${siteUrl}/#software` },
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          {
            "@type": "ListItem",
            position: 1,
            name: "CourtFlux",
            item: siteUrl,
          },
          {
            "@type": "ListItem",
            position: 2,
            name: "Basketball Video Analysis",
            item: pageUrl,
          },
        ],
      },
      {
        "@type": "FAQPage",
        mainEntity: faq.map(([question, answer]) => ({
          "@type": "Question",
          name: question,
          acceptedAnswer: {
            "@type": "Answer",
            text: answer,
          },
        })),
      },
    ],
  };

  return (
    <main className={styles.page}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData) }}
      />

      <header className={styles.header}>
        <Link href="/" aria-label="CourtFlux home" className={styles.logoLink}>
          <Image
            src="/brand/horizontal_logo_safe_crop.png"
            alt="CourtFlux — Basketball Intelligence"
            width={660}
            height={185}
            className={styles.logo}
            priority
          />
        </Link>
        <nav className={styles.nav} aria-label="Basketball video analysis navigation">
          <a href="#how-it-works">How it works</a>
          <a href="#capabilities">Capabilities</a>
          <Link href="/#pricing">Pricing</Link>
          <Link href="/login">Sign in</Link>
          <Link href="/login?mode=signup" className={styles.navCta}>Start free beta</Link>
        </nav>
      </header>

      <div className={styles.shell}>
        <section className={styles.hero}>
          <div className={styles.heroCopy}>
            <div className={styles.eyebrow}>Basketball video analysis software</div>
            <h1>Basketball video analysis that starts with <span>the film you already have.</span></h1>
            <p className={styles.lead}>
              CourtFlux is being built to turn ordinary basketball game footage into structured player, team and ball evidence so coaches can spend less time searching video and more time interpreting the game.
            </p>
            <div className={styles.actions}>
              <Link href="/login?mode=signup" className={styles.primaryButton}>Start free beta</Link>
              <a href="#how-it-works" className={styles.secondaryButton}>See how it works</a>
            </div>
            <div className={styles.trustRow} aria-label="CourtFlux product principles">
              <span>No proprietary camera required</span>
              <span>Existing game film</span>
              <span>Coach-first workflow</span>
            </div>
          </div>

          <div className={styles.heroVisual}>
            <Image
              src="/brand/CourtFlux_Full_Analyst_Workspace.png"
              alt="CourtFlux basketball video analysis workspace showing game footage, tracked players, ball movement and analysis panels"
              width={1672}
              height={941}
              sizes="(max-width: 920px) 100vw, 55vw"
              priority
            />
          </div>
        </section>

        <section className={styles.problemStrip} aria-label="Basketball video analysis workflow">
          <article><strong>Record the game</strong><span>Start with basketball footage you already capture.</span></article>
          <article><strong>Track what happens</strong><span>Build temporal context around players, teams and the ball.</span></article>
          <article><strong>Find useful evidence</strong><span>Move toward searchable basketball moments instead of manual scrubbing.</span></article>
        </section>

        <section id="how-it-works" className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>How CourtFlux approaches video analysis</div>
            <h2>From pixels to <span>basketball context.</span></h2>
            <p>
              Useful basketball analysis depends on more than detecting a player in one frame. CourtFlux is structured around temporal continuity: what is on screen, where it moves, what persists, and how those observations can later support possessions, events and tactical interpretation.
            </p>
          </div>

          <div className={styles.twoColumn}>
            <div className={styles.imageFrame}>
              <Image
                src="/brand/CourtFlux_Video_Tracking_Tactical_Insight.png"
                alt="CourtFlux workflow showing basketball video progressing from object tracking toward tactical insight"
                width={768}
                height={1024}
                sizes="(max-width: 920px) 100vw, 46vw"
              />
            </div>
            <div className={styles.steps}>
              <article>
                <div className={styles.stepNumber}>01</div>
                <h3>Video becomes tracked objects.</h3>
                <p>Player detections, team classification and ball candidates create the first structured layer above raw footage.</p>
              </article>
              <article>
                <div className={styles.stepNumber}>02</div>
                <h3>Frames become trajectories.</h3>
                <p>Tracking across time matters because basketball decisions happen through movement, occlusion, transitions and changes of possession.</p>
              </article>
              <article>
                <div className={styles.stepNumber}>03</div>
                <h3>Trajectories become basketball evidence.</h3>
                <p>As the product develops, spatial mapping, possessions, events and tactical patterns can build on a stronger tracking foundation.</p>
              </article>
            </div>
          </div>
        </section>

        <section id="capabilities" className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>What is available vs what is coming</div>
            <h2>A basketball analysis stack built <span>in the right order.</span></h2>
            <p>
              CourtFlux does not present future basketball intelligence as if it were already production-ready. The current beta concentrates on the computer-vision foundation required for later structured analytics.
            </p>
          </div>

          <div className={styles.statusGrid}>
            <article className={styles.statusCard}>
              <div className={styles.statusLabel}>Current beta focus</div>
              <h3>Tracking foundation</h3>
              <ul>
                <li>Player detection and tracking</li>
                <li>Team classification</li>
                <li>Ball detection</li>
                <li>Temporal ball tracking</li>
                <li>Processing diagnostics</li>
              </ul>
            </article>
            <article className={styles.statusCard}>
              <div className={styles.statusLabel}>In development</div>
              <h3>Basketball structure</h3>
              <ul>
                <li>Player identity and attribution</li>
                <li>Court mapping</li>
                <li>Possession segmentation</li>
                <li>Basketball event detection</li>
                <li>Structured player and team analytics</li>
              </ul>
            </article>
            <article className={styles.statusCard}>
              <div className={styles.statusLabel}>Product vision</div>
              <h3>Coaching intelligence</h3>
              <ul>
                <li>Spacing and defensive shape</li>
                <li>Actions and tactical patterns</li>
                <li>Player development intelligence</li>
                <li>Coach-facing interpretation</li>
                <li>High-performance workflows</li>
              </ul>
            </article>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.twoColumnReverse}>
            <div>
              <div className={styles.eyebrow}>Ball intelligence</div>
              <h2>The ball is small on screen but <span>central to the game.</span></h2>
              <p className={styles.bodyCopy}>
                Possessions, passes, shots, rebounds and turnovers all depend on knowing where the ball is and maintaining continuity when it is briefly hidden. CourtFlux combines direct ball detections with temporal association and conservative short-gap prediction rather than treating every frame independently.
              </p>
              <p className={styles.note}>Ball quality remains an active benchmark area. CourtFlux does not claim perfect tracking across every camera condition.</p>
            </div>
            <div className={styles.imageFrame}>
              <Image
                src="/brand/CourtFlux_Ball_Intelligence.png"
                alt="CourtFlux ball intelligence visualization with basketball ball detection and temporal trajectory tracking"
                width={768}
                height={1024}
                sizes="(max-width: 920px) 100vw, 46vw"
              />
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>Who it is for</div>
            <h2>One analysis foundation. <span>Different basketball workflows.</span></h2>
          </div>
          <div className={styles.audienceGrid}>
            <article>
              <h3>Coaches</h3>
              <p>Reduce repetitive video searching and build a clearer evidence trail around the moments you want to teach.</p>
            </article>
            <article>
              <h3>Teams</h3>
              <p>Build a season-long library where staff can review the same game context, tracked evidence and future structured analytics.</p>
            </article>
            <article>
              <h3>Academies</h3>
              <p>Extend the same analysis model across multiple squads and, over time, connect game evidence to longitudinal player development.</p>
            </article>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>FAQ</div>
            <h2>Questions about <span>basketball video analysis.</span></h2>
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
          <div>
            <div className={styles.eyebrow}>CourtFlux private beta</div>
            <h2>Turn basketball video into a <span>better analysis starting point.</span></h2>
            <p>Start with the footage you already have and help shape the tracking and analysis workflow as CourtFlux develops.</p>
          </div>
          <div className={styles.actions}>
            <Link href="/login?mode=signup" className={styles.primaryButton}>Start free beta</Link>
            <Link href="/#pricing" className={styles.secondaryButton}>See pricing</Link>
          </div>
        </section>

        <footer className={styles.footer}>
          <Link href="/" aria-label="CourtFlux home">
            <Image
              src="/brand/horizontal_logo_safe_crop.png"
              alt="CourtFlux — Basketball Intelligence"
              width={660}
              height={185}
              className={styles.footerLogo}
            />
          </Link>
          <div className={styles.footerLinks}>
            <Link href="/">Home</Link>
            <Link href="/#pricing">Pricing</Link>
            <Link href="/login">Sign in</Link>
          </div>
        </footer>
      </div>
    </main>
  );
}
