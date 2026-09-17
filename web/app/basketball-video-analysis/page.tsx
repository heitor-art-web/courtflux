import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import englishContent from "@/data/pages/basketball-video-analysis.json";
import { localizePublicHref } from "@/lib/i18n/config";
import { localizeContent } from "@/lib/i18n/public-content";
import { getRequestLocale } from "@/lib/i18n/request";
import { metadataFromContent } from "@/lib/page-content";
import { getSiteUrl } from "@/lib/site";
import styles from "./page.module.css";

const siteUrl = getSiteUrl();

export async function generateMetadata(): Promise<Metadata> {
  const locale = await getRequestLocale();
  return metadataFromContent(localizeContent(englishContent, locale).metadata);
}

export default async function BasketballVideoAnalysisPage() {
  const locale = await getRequestLocale();
  const content = localizeContent(englishContent, locale);
  const path = content.metadata.canonical;
  const pageUrl = `${siteUrl}${path}`;
  const homeHref = localizePublicHref("/", locale);
  const structuredData = {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": `${pageUrl}#webpage`,
        url: pageUrl,
        name: content.schema.pageName,
        description: content.schema.pageDescription,
        isPartOf: { "@id": `${siteUrl}/#website` },
        about: { "@id": `${siteUrl}/#software` },
        inLanguage: locale,
      },
      {
        "@type": "BreadcrumbList",
        itemListElement: [
          { "@type": "ListItem", position: 1, name: "CourtFlux", item: `${siteUrl}${homeHref === "/" ? "" : homeHref}` },
          { "@type": "ListItem", position: 2, name: content.schema.breadcrumbName, item: pageUrl },
        ],
      },
      {
        "@type": "FAQPage",
        inLanguage: locale,
        mainEntity: content.faq.map(({ question, answer }) => ({
          "@type": "Question",
          name: question,
          acceptedAnswer: { "@type": "Answer", text: answer },
        })),
      },
    ],
  };

  return (
    <main className={styles.page}>
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData).replace(/</g, "\\u003c") }} />

      <header className={styles.header}>
        <Link href={homeHref} aria-label="CourtFlux home" className={styles.logoLink}>
          <Image src="/brand/light_logo_safe_crop.png" alt="CourtFlux — Basketball Intelligence" width={320} height={255} className={styles.logo} priority />
        </Link>
        <nav className={styles.nav} aria-label="Basketball video analysis navigation">
          {content.navigation.map((item) => (
            <Link href={item.href} className={item.cta ? styles.navCta : undefined} key={item.href}>{item.label}</Link>
          ))}
        </nav>
      </header>

      <div className={styles.shell}>
        <section className={styles.hero}>
          <div className={styles.heroCopy}>
            <div className={styles.eyebrow}>{content.hero.eyebrow}</div>
            <h1>{content.hero.title} <span>{content.hero.highlight}</span></h1>
            <p className={styles.lead}>{content.hero.lead}</p>
            <div className={styles.actions}>
              <Link href={content.hero.primary.href} className={styles.primaryButton}>{content.hero.primary.label}</Link>
              <Link href={content.hero.secondary.href} className={styles.secondaryButton}>{content.hero.secondary.label}</Link>
            </div>
            <div className={styles.trustRow} aria-label="CourtFlux product principles">
              {content.hero.trust.map((item) => <span key={item}>{item}</span>)}
            </div>
          </div>

          <div className={styles.heroVisual}>
            <Image src={content.hero.image.src} alt={content.hero.image.alt} width={content.hero.image.width} height={content.hero.image.height} sizes="(max-width: 920px) 100vw, 55vw" priority />
          </div>
        </section>

        <section className={styles.problemStrip} aria-label="Basketball video analysis workflow">
          {content.problemStrip.map((item) => <article key={item.title}><strong>{item.title}</strong><span>{item.copy}</span></article>)}
        </section>

        <section id="how-it-works" className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>{content.how.eyebrow}</div>
            <h2>{content.how.title} <span>{content.how.highlight}</span></h2>
            <p>{content.how.copy}</p>
          </div>

          <div className={styles.twoColumn}>
            <div className={styles.imageFrame}>
              <Image src={content.how.image.src} alt={content.how.image.alt} width={content.how.image.width} height={content.how.image.height} sizes="(max-width: 920px) 100vw, 46vw" />
            </div>
            <div className={styles.steps}>
              {content.how.steps.map((step) => (
                <article key={step.number}>
                  <div className={styles.stepNumber}>{step.number}</div>
                  <h3>{step.title}</h3>
                  <p>{step.copy}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="capabilities" className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>{content.capabilities.eyebrow}</div>
            <h2>{content.capabilities.title} <span>{content.capabilities.highlight}</span></h2>
            <p>{content.capabilities.copy}</p>
          </div>

          <div className={styles.statusGrid}>
            {content.capabilities.cards.map((card) => (
              <article className={styles.statusCard} key={card.title}>
                <div className={styles.statusLabel}>{card.label}</div>
                <h3>{card.title}</h3>
                <ul>{card.items.map((item) => <li key={item}>{item}</li>)}</ul>
              </article>
            ))}
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.twoColumnReverse}>
            <div>
              <div className={styles.eyebrow}>{content.ball.eyebrow}</div>
              <h2>{content.ball.title} <span>{content.ball.highlight}</span></h2>
              <p className={styles.bodyCopy}>{content.ball.copy}</p>
              <p className={styles.note}>{content.ball.note}</p>
            </div>
            <div className={styles.imageFrame}>
              <Image src={content.ball.image.src} alt={content.ball.image.alt} width={content.ball.image.width} height={content.ball.image.height} sizes="(max-width: 920px) 100vw, 46vw" />
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>{content.audiences.eyebrow}</div>
            <h2>{content.audiences.title} <span>{content.audiences.highlight}</span></h2>
          </div>
          <div className={styles.audienceGrid}>
            {content.audiences.cards.map((audience) => (
              <article key={audience.title}>
                <h3>{audience.title}</h3>
                <p>{audience.copy}</p>
                {"href" in audience && audience.href ? <Link href={audience.href}>{audience.linkLabel} →</Link> : <span>{audience.linkLabel}</span>}
              </article>
            ))}
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionIntro}>
            <div className={styles.eyebrow}>{content.faqIntro.eyebrow}</div>
            <h2>{content.faqIntro.title} <span>{content.faqIntro.highlight}</span></h2>
          </div>
          <div className={styles.faqGrid}>
            {content.faq.map(({ question, answer }) => <details key={question}><summary>{question}</summary><p>{answer}</p></details>)}
          </div>
        </section>

        <section className={styles.finalCta}>
          <div>
            <div className={styles.eyebrow}>{content.finalCta.eyebrow}</div>
            <h2>{content.finalCta.title} <span>{content.finalCta.highlight}</span></h2>
            <p>{content.finalCta.copy}</p>
          </div>
          <div className={styles.actions}>
            <Link href={content.finalCta.primary.href} className={styles.primaryButton}>{content.finalCta.primary.label}</Link>
            <Link href={content.finalCta.secondary.href} className={styles.secondaryButton}>{content.finalCta.secondary.label}</Link>
          </div>
        </section>

        <footer className={styles.footer}>
          <Link href={homeHref} aria-label="CourtFlux home">
            <Image src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux — Basketball Intelligence" width={660} height={185} className={styles.footerLogo} />
          </Link>
          <div className={styles.footerLinks}>{content.footer.map((item) => <Link href={item.href} key={`${item.href}-${item.label}`}>{item.label}</Link>)}</div>
        </footer>
      </div>
    </main>
  );
}
