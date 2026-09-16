import Image from "next/image";
import Link from "next/link";
import type {
  AudienceSection,
  EvidenceSection,
  FaqSection,
  FinalCtaSection,
  HeroSection,
  HomeSection,
  HowSection,
  PricingSection,
  StorySection,
  ValueStripSection,
} from "@/lib/home-content";
import styles from "./home.module.css";
import pricingStyles from "./pricing-v2.module.css";

function SectionHeading({ eyebrow, title, highlight, copy }: { eyebrow: string; title: string; highlight: string; copy?: string }) {
  return (
    <div className={styles.sectionHeader}>
      <div className={styles.eyebrow}>{eyebrow}</div>
      <h2 className={styles.sectionTitle}>{title} <span>{highlight}</span></h2>
      {copy ? <p className={styles.sectionCopy}>{copy}</p> : null}
    </div>
  );
}

function Hero({ section }: { section: HeroSection }) {
  return (
    <section id={section.id} className={styles.hero} aria-labelledby="hero-title">
      <div className={styles.heroContent}>
        <div className={styles.eyebrow}>{section.eyebrow}</div>
        <h1 id="hero-title">{section.title} <span>{section.highlight}</span></h1>
        <p className={styles.heroCopy}>{section.copy}</p>
        <div className={styles.actions}>
          <Link className={styles.primaryButton} href={section.primary.href}>{section.primary.label}</Link>
          <Link className={styles.secondaryButton} href={section.secondary.href}>{section.secondary.label}</Link>
        </div>
        <div className={styles.pills} aria-label={section.eyebrow}>
          {section.pills.map((pill) => <span className={styles.pill} key={pill}>{pill}</span>)}
        </div>
      </div>

      <div className={styles.heroVisual}>
        <div className={styles.visualFrame}>
          <Image
            src={section.media.src}
            alt={section.media.alt}
            width={section.media.width}
            height={section.media.height}
            sizes="(max-width: 900px) 100vw, 58vw"
            priority
          />
        </div>
        <div className={styles.heroInsight}>
          <span>{section.overlay.kicker}</span>
          <strong>{section.overlay.title}</strong>
          <div className={styles.heroMetrics}>
            {section.overlay.metrics.map((metric) => <em key={metric}>{metric}</em>)}
          </div>
          <small>{section.overlay.note}</small>
        </div>
      </div>
    </section>
  );
}

function ValueStrip({ section }: { section: ValueStripSection }) {
  return (
    <section id={section.id} className={styles.valueStrip}>
      {section.items.map((item, index) => (
        <article className={styles.valueItem} key={item.title}>
          <span>0{index + 1}</span>
          <div><strong>{item.title}</strong><p>{item.copy}</p></div>
        </article>
      ))}
    </section>
  );
}

function Story({ section }: { section: StorySection }) {
  const text = (
    <div className={styles.storyCopy}>
      <div className={styles.eyebrow}>{section.eyebrow}</div>
      <h2 className={styles.sectionTitle}>{section.title} <span>{section.highlight}</span></h2>
      <p className={styles.sectionCopy}>{section.copy}</p>
      {section.points?.length ? (
        <div className={styles.storyPoints}>
          {section.points.map((point) => (
            <div className={styles.storyPoint} key={point.title}>
              <strong>{point.title}</strong>
              <p>{point.copy}</p>
            </div>
          ))}
        </div>
      ) : null}
      {section.note ? <p className={styles.smallNote}>{section.note}</p> : null}
    </div>
  );

  const media = (
    <div className={styles.visualFrame}>
      <Image
        src={section.media.src}
        alt={section.media.alt}
        width={section.media.width}
        height={section.media.height}
        sizes="(max-width: 900px) 100vw, 52vw"
      />
    </div>
  );

  return (
    <section id={section.id} className={styles.section}>
      <div className={section.variant === "media-left" ? styles.storyGrid : `${styles.storyGrid} ${styles.storyGridReverse}`}>
        {section.variant === "media-left" ? <>{media}{text}</> : <>{text}{media}</>}
      </div>
    </section>
  );
}

function Evidence({ section }: { section: EvidenceSection }) {
  return (
    <section id={section.id} className={`${styles.section} ${styles.evidenceSection}`}>
      <SectionHeading eyebrow={section.eyebrow} title={section.title} highlight={section.highlight} copy={section.copy} />
      <div className={styles.evidenceGrid}>
        <div className={styles.visualFrame}>
          <Image
            src={section.media.src}
            alt={section.media.alt}
            width={section.media.width}
            height={section.media.height}
            sizes="(max-width: 900px) 100vw, 58vw"
          />
        </div>
        <article className={styles.insightCard}>
          <span className={styles.kicker}>{section.insight.kicker}</span>
          <strong>{section.insight.title}</strong>
          <p>{section.insight.copy}</p>
          <div className={styles.insightAction}>{section.insight.action} <span aria-hidden="true">→</span></div>
        </article>
      </div>
    </section>
  );
}

function Audiences({ section }: { section: AudienceSection }) {
  return (
    <section id={section.id} className={styles.section}>
      <SectionHeading eyebrow={section.eyebrow} title={section.title} highlight={section.highlight} copy={section.copy} />
      <div className={styles.audienceGrid}>
        {section.items.map((item) => (
          <Link className={styles.audienceCard} href={item.href} key={item.audience}>
            <span>{item.audience}</span>
            <strong>{item.title}</strong>
            <p>{item.copy}</p>
            <em>{section.actionLabel} <span aria-hidden="true">→</span></em>
          </Link>
        ))}
      </div>
    </section>
  );
}

function HowItWorks({ section }: { section: HowSection }) {
  return (
    <section id={section.id} className={`${styles.section} ${styles.howSection}`}>
      <SectionHeading eyebrow={section.eyebrow} title={section.title} highlight={section.highlight} copy={section.copy} />
      <div className={styles.howGrid}>
        {section.steps.map((step) => (
          <article className={styles.howStep} key={step.number}>
            <span>{step.number}</span>
            <strong>{step.title}</strong>
            <p>{step.copy}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function Pricing({ section }: { section: PricingSection }) {
  return (
    <section id={section.id} className={styles.section}>
      <SectionHeading eyebrow={section.eyebrow} title={section.title} highlight={section.highlight} copy={section.copy} />

      <div className={pricingStyles.trialBanner}>
        <div>
          <span className={styles.kicker}>{section.trial.kicker}</span>
          <strong>{section.trial.title}</strong>
          <p>{section.trial.copy}</p>
          <small>{section.trial.meta}</small>
        </div>
        <Link className={styles.primaryButton} href={section.trial.cta.href}>{section.trial.cta.label}</Link>
      </div>

      <div className={`${styles.pricingGrid} ${pricingStyles.pricingGrid}`}>
        {section.plans.map((plan) => (
          <article className={`${styles.priceCard} ${pricingStyles.priceCard} ${plan.featured ? styles.priceCardFeatured : ""}`} key={plan.name}>
            {plan.tag ? <span className={styles.priceTag}>{plan.tag}</span> : null}
            <div className={styles.kicker}>{plan.kicker}</div>
            <h3>{plan.name}</h3>
            <div className={styles.price}>{plan.price} <small>{plan.suffix}</small></div>
            <div className={pricingStyles.annualPrice}>{plan.annualPrice} <span>{plan.annualSuffix}</span></div>
            <p className={styles.priceDescription}>{plan.description}</p>
            <div className={styles.usage}><strong>{plan.allowance}</strong><span>{plan.allowanceCopy}</span></div>
            <ul className={styles.featureList}>{plan.features.map((feature) => <li key={feature}>{feature}</li>)}</ul>
            <Link className={plan.featured ? styles.primaryButton : styles.secondaryButton} href={plan.href}>{plan.cta}</Link>
          </article>
        ))}
      </div>

      <div className={pricingStyles.organisationRow}>
        <div>
          <strong>{section.organisation.title}</strong>
          <p>{section.organisation.copy}</p>
        </div>
        <Link className={styles.secondaryButton} href={section.organisation.cta.href}>{section.organisation.cta.label}</Link>
      </div>

      <p className={styles.pricingNote}>{section.note}</p>
    </section>
  );
}

function Faq({ section }: { section: FaqSection }) {
  return (
    <section id={section.id} className={styles.section}>
      <SectionHeading eyebrow={section.eyebrow} title={section.title} highlight={section.highlight} />
      <div className={styles.faqGrid}>
        {section.items.map(({ question, answer }) => (
          <details key={question}><summary>{question}</summary><p>{answer}</p></details>
        ))}
      </div>
    </section>
  );
}

function FinalCta({ section }: { section: FinalCtaSection }) {
  return (
    <section id={section.id} className={styles.finalCta}>
      <div className={styles.eyebrow}>{section.eyebrow}</div>
      <h2 className={styles.sectionTitle}>{section.title} <span>{section.highlight}</span></h2>
      <p className={styles.sectionCopy}>{section.copy}</p>
      <div className={styles.actions}>
        <Link className={styles.primaryButton} href={section.primary.href}>{section.primary.label}</Link>
        <Link className={styles.secondaryButton} href={section.secondary.href}>{section.secondary.label}</Link>
      </div>
    </section>
  );
}

export function renderHomeSection(section: HomeSection) {
  switch (section.type) {
    case "hero": return <Hero key={section.id ?? section.type} section={section} />;
    case "valueStrip": return <ValueStrip key={section.id ?? section.type} section={section} />;
    case "story": return <Story key={section.id ?? `${section.type}-${section.eyebrow}`} section={section} />;
    case "evidence": return <Evidence key={section.id ?? section.type} section={section} />;
    case "audiences": return <Audiences key={section.id ?? section.type} section={section} />;
    case "how": return <HowItWorks key={section.id ?? section.type} section={section} />;
    case "pricing": return <Pricing key={section.id ?? section.type} section={section} />;
    case "faq": return <Faq key={section.id ?? section.type} section={section} />;
    case "finalCta": return <FinalCta key={section.id ?? section.type} section={section} />;
  }
}
