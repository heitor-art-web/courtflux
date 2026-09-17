import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import englishContent from "@/data/pages/home.json";
import { getFaqSection, type HomeContent } from "@/lib/home-content";
import { localizePublicHref } from "@/lib/i18n/config";
import { localizeContent } from "@/lib/i18n/public-content";
import { getRequestLocale } from "@/lib/i18n/request";
import { metadataFromContent } from "@/lib/page-content";
import { getSiteUrl } from "@/lib/site";
import { renderHomeSection } from "./home-sections";
import styles from "./home.module.css";
import polish from "./home-polish.module.css";

const siteUrl = getSiteUrl();
const sourceContent = englishContent as HomeContent;

export async function generateMetadata(): Promise<Metadata> {
  const locale = await getRequestLocale();
  return metadataFromContent(localizeContent(sourceContent, locale).metadata);
}

export default async function HomePage() {
  const locale = await getRequestLocale();
  const content = localizeContent(sourceContent, locale);
  const faq = getFaqSection(content);
  const homeHref = localizePublicHref("/", locale);
  const localizedSiteUrl = `${siteUrl}${homeHref === "/" ? "" : homeHref}`;
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
        inLanguage: locale,
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
        "@id": `${localizedSiteUrl}#webpage`,
        url: localizedSiteUrl,
        name: content.schema.webPageName,
        description: content.schema.webPageDescription,
        isPartOf: { "@id": `${siteUrl}/#website` },
        about: { "@id": `${siteUrl}/#software` },
        inLanguage: locale,
      },
      ...(faq ? [{
        "@type": "FAQPage",
        "@id": `${localizedSiteUrl}#faq`,
        inLanguage: locale,
        mainEntity: faq.items.map(({ question, answer }) => ({
          "@type": "Question",
          name: question,
          acceptedAnswer: { "@type": "Answer", text: answer },
        })),
      }] : []),
    ],
  };

  return (
    <main className={styles.home}>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(structuredData).replace(/</g, "\\u003c") }}
      />

      <header className={`${styles.siteHeader} ${polish.header}`}>
        <Link href={homeHref} className={`${styles.logoLink} ${polish.logoPanel}`} aria-label="CourtFlux home">
          <Image
            className={`${styles.logo} ${polish.headerLogo}`}
            src="/brand/light_logo_safe_crop.png"
            alt="CourtFlux — Basketball Intelligence"
            width={320}
            height={255}
            priority
          />
        </Link>
        <nav className={`${styles.nav} ${polish.nav}`} aria-label="Primary navigation">
          {content.navigation.map((item) => (
            <Link className={item.cta ? styles.navCta : undefined} href={item.href} key={`${item.href}-${item.label}`}>
              {item.label}
            </Link>
          ))}
        </nav>
      </header>

      <div className={styles.shell}>
        {content.sections.map(renderHomeSection)}

        <footer className={`${styles.footer} ${polish.footer}`}>
          <div className={styles.footerInner}>
            <div>
              <div className={polish.footerBrandPanel}>
                <Image className={`${styles.footerLogo} ${polish.footerLogo}`} src="/brand/light_logo_safe_crop.png" alt="CourtFlux — Basketball Intelligence" width={320} height={255} />
              </div>
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
