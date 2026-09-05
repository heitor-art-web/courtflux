import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import content from "@/data/pages/home.json";
import { metadataFromContent } from "@/lib/page-content";
import { getSiteUrl } from "@/lib/site";
import styles from "./home.module.css";

const siteUrl = getSiteUrl();

export const metadata: Metadata = metadataFromContent(content.metadata);

const toneClasses = {
  now: styles.now,
  doing: styles.doing,
  later: styles.later,
};

function CapabilityCard({ item }: { item: (typeof content.capabilities)[number] }) {
  return (
    <article className={styles.card}>
      <div className={styles.kicker}>{item.kicker}</div>
      <h3>{item.title}</h3>
      <p>{item.copy}</p>
      <span className={styles.badge}>{item.status}</span>
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
        description: content.schema.organizationDescription,
      },
      {
        "@type": "WebSite",
        "@id": `${siteUrl}/#website`,
        name: "CourtFlux",
        url: siteUrl,
        publisher: { "@id": `${siteUrl}/#organization` },
        inLanguage: "en",
      },
      {
        "@type": "SoftwareApplication",
        "@id": `${siteUrl}/#software`,
        name: "CourtFlux",
        url: siteUrl,
        applicationCategory: "SportsApplication",
        operatingSystem: "Web",
        description: content.schema.softwareDescription,
        featureList: content.schema.featureList,
        audience: {
          "@type": "Audience",
          audienceType: content.schema.audienceType,
        },
        publisher: { "@id": `${siteUrl}/#organization` },
      },
      {
        "@type": "WebPage",
        "@id": `${siteUrl}/#webpage`,
        url: siteUrl,
        name: content.schema.webPageName,
        description: content.schema.webPageDescription,
        isPartOf: { "@id": `${siteUrl}/#website` },
        about: { "@id": `${siteUrl}/#software` },
        inLanguage: "en",
      },
      {
        "@type": "FAQPage",
        "@id": `${siteUrl}/#faq`,
        mainEntity: content.faq.map(({ question, answer }) => ({
          "@type": "Question",
          name: question,
          acceptedAnswer: { "@type": "Answer", text: answer },
        })),
      },
    ],
  };

  return (
    <main className={styles.home}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData).replace(/</g, "\\u003c") }}
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
          {content.navigation.map((item) => (
            <Link className={item.cta ? styles.navCta : undefined} href={item.href} key={item.href}>
              {item.label}
            </Link>
          ))}
        </nav>
      </header>

      <div className={styles.shell}>
        <section className={styles.hero} aria-labelledby="hero-title">
          <div>
            <div className={styles.eyebrow}>{content.hero.eyebrow}</div>
            <h1 id="hero-title">{content.hero.title} <span>{content.hero.highlight}</span></h1>
            <p className={styles.heroCopy}>{content.hero.copy}</p>
            <div className={styles.actions}>
              <Link className={styles.primaryButton} href={content.hero.primary.href}>{content.hero.primary.label}</Link>
              <Link className={styles.secondaryButton} href={content.hero.secondary.href}>{content.hero.secondary.label}</Link>
            </div>
            <div className={styles.pills} aria-label="Product principles">
              {content.hero.pills.map((pill) => <span className={styles.pill} key={pill}>{pill}</span>)}
            </div>
          </div>

          <div className={styles.visualFrame}>
            <Image
              src={content.hero.image.src}
              alt={content.hero.image.alt}
              width={content.hero.image.width}
              height={content.hero.image.height}
              sizes="(max-width: 900px) 100vw, 58vw"
              priority
            />
          </div>
        </section>

        <section className={styles.signalGrid} aria-label="CourtFlux workflow">
          {content.signals.map((signal) => (
            <div className={styles.signal} key={signal.title}><strong>{signal.title}</strong><span>{signal.copy}</span></div>
          ))}
        </section>

        <section id="how" className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>{content.how.eyebrow}</div>
            <h2 className={styles.sectionTitle}>{content.how.title} <span>{content.how.highlight}</span></h2>
            <p className={styles.sectionCopy}>{content.how.copy}</p>
          </div>

          <div className={styles.split}>
            <div className={styles.visualFrame}>
              <Image
                src={content.how.image.src}
                alt={content.how.image.alt}
                width={content.how.image.width}
                height={content.how.image.height}
                sizes="(max-width: 900px) 100vw, 52vw"
              />
            </div>
            <div>
              <div className={styles.stack}>
                {content.how.steps.map((step) => (
                  <article className={styles.step} key={step.kicker}>
                    <div className={styles.kicker}>{step.kicker}</div>
                    <h3>{step.title}</h3>
                    <p>{step.copy}</p>
                  </article>
                ))}
              </div>
              <div className={styles.contextBox}>
                <strong>{content.how.context.title}</strong>
                <p>{content.how.context.copy}</p>
              </div>
            </div>
          </div>
        </section>

        <section id="product" className={styles.section}>
          <div className={`${styles.splitReverse} ${styles.split}`}>
            <div>
              <div className={styles.eyebrow}>{content.ball.eyebrow}</div>
              <h2 className={styles.sectionTitle}>{content.ball.title} <span>{content.ball.highlight}</span></h2>
              <p className={styles.sectionCopy}>{content.ball.copy}</p>
              <ul className={styles.bulletList}>
                {content.ball.bullets.map((item) => <li key={item.title}><strong>{item.title}</strong> — {item.copy}</li>)}
              </ul>
            </div>
            <div className={styles.visualFrame}>
              <Image
                src={content.ball.image.src}
                alt={content.ball.image.alt}
                width={content.ball.image.width}
                height={content.ball.image.height}
                sizes="(max-width: 900px) 100vw, 48vw"
              />
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.split}>
            <div className={styles.visualFrame}>
              <Image
                src={content.spatial.image.src}
                alt={content.spatial.image.alt}
                width={content.spatial.image.width}
                height={content.spatial.image.height}
                sizes="(max-width: 900px) 100vw, 54vw"
              />
            </div>
            <div>
              <div className={styles.eyebrow}>{content.spatial.eyebrow}</div>
              <h2 className={styles.sectionTitle}>{content.spatial.title} <span>{content.spatial.highlight}</span></h2>
              <p className={styles.sectionCopy}>{content.spatial.copy}</p>
              <p className={styles.smallNote}>{content.spatial.note}</p>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>{content.capabilitiesIntro.eyebrow}</div>
            <h2 className={styles.sectionTitle}>{content.capabilitiesIntro.title} <span>{content.capabilitiesIntro.highlight}</span></h2>
            <p className={styles.sectionCopy}>{content.capabilitiesIntro.copy}</p>
          </div>
          <div className={styles.cardGrid}>
            {content.capabilities.map((item) => <CapabilityCard key={item.title} item={item} />)}
          </div>
        </section>

        <section id="pricing" className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>{content.pricing.intro.eyebrow}</div>
            <h2 className={styles.sectionTitle}>{content.pricing.intro.title} <span>{content.pricing.intro.highlight}</span></h2>
            <p className={styles.sectionCopy}>{content.pricing.intro.copy}</p>
          </div>

          <div className={styles.pricingGrid}>
            {content.pricing.plans.map((plan) => (
              <article className={`${styles.priceCard} ${plan.featured ? styles.priceCardFeatured : ""}`} key={plan.name}>
                {"tag" in plan && plan.tag ? <span className={styles.priceTag}>{plan.tag}</span> : null}
                <div className={styles.kicker}>{plan.kicker}</div>
                <h3>{plan.name}</h3>
                <div className={styles.price}>{plan.price} <small>{plan.suffix}</small></div>
                <p className={styles.priceDescription}>{plan.description}</p>
                <div className={styles.usage}><strong>{plan.allowance}</strong><span>{plan.allowanceCopy}</span></div>
                <ul className={styles.featureList}>{plan.features.map((feature) => <li key={feature}>{feature}</li>)}</ul>
                <Link className={plan.featured ? styles.primaryButton : styles.secondaryButton} href={plan.href}>{plan.cta}</Link>
              </article>
            ))}
          </div>

          <div className={styles.usageExplainer}>
            <div>
              <strong>{content.pricing.usageExplainer.title}</strong>
              <p>{content.pricing.usageExplainer.copy}</p>
            </div>
            <div>
              <div className={styles.usageScale}>{content.pricing.usageExplainer.scale.map((item) => <span key={item}>{item}</span>)}</div>
              <p>{content.pricing.usageExplainer.scaleCopy}</p>
            </div>
          </div>
          <p className={styles.pricingNote}>{content.pricing.note}</p>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>{content.transparency.intro.eyebrow}</div>
            <h2 className={styles.sectionTitle}>{content.transparency.intro.title} <span>{content.transparency.intro.highlight}</span></h2>
            <p className={styles.sectionCopy}>{content.transparency.intro.copy}</p>
          </div>
          <div className={styles.statusGrid}>
            {content.transparency.columns.map((column) => (
              <article className={styles.statusCard} key={column.kicker}>
                <div className={styles.kicker}>{column.kicker}</div>
                <ul>{column.items.map((item) => <li key={item}><span className={toneClasses[column.tone as keyof typeof toneClasses]}>●</span> {item}</li>)}</ul>
              </article>
            ))}
          </div>

          <div className={styles.betaCta}>
            <div className={styles.eyebrow}>{content.betaLab.eyebrow}</div>
            <h3>{content.betaLab.title}</h3>
            <p>{content.betaLab.copy}</p>
            <div className={styles.actions}><Link className={styles.secondaryButton} href={content.betaLab.cta.href}>{content.betaLab.cta.label}</Link></div>
          </div>
        </section>

        <section className={styles.section} id="faq">
          <div className={styles.sectionHeader}>
            <div className={styles.eyebrow}>{content.faqIntro.eyebrow}</div>
            <h2 className={styles.sectionTitle}>{content.faqIntro.title} <span>{content.faqIntro.highlight}</span></h2>
          </div>
          <div className={styles.faqGrid}>
            {content.faq.map(({ question, answer }) => (
              <details key={question}><summary>{question}</summary><p>{answer}</p></details>
            ))}
          </div>
        </section>

        <section className={styles.finalCta}>
          <div className={styles.eyebrow}>{content.finalCta.eyebrow}</div>
          <h2 className={styles.sectionTitle}>{content.finalCta.title} <span>{content.finalCta.highlight}</span></h2>
          <p className={styles.sectionCopy}>{content.finalCta.copy}</p>
          <div className={styles.actions}>
            <Link className={styles.primaryButton} href={content.finalCta.primary.href}>{content.finalCta.primary.label}</Link>
            <Link className={styles.secondaryButton} href={content.finalCta.secondary.href}>{content.finalCta.secondary.label}</Link>
          </div>
        </section>

        <footer className={styles.footer}>
          <div className={styles.footerInner}>
            <div>
              <Image className={styles.footerLogo} src="/brand/horizontal_logo_safe_crop.png" alt="CourtFlux — Basketball Intelligence" width={660} height={185} />
              <div className={styles.footerMeta}>{content.footer.meta}</div>
            </div>
            <div className={styles.footerLinks}>
              {content.footer.links.map((item) => <Link href={item.href} key={`${item.href}-${item.label}`}>{item.label}</Link>)}
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
