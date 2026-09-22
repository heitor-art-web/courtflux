import type { Metadata } from "next";
import englishContent from "@/data/pages/home.json";
import { getFaqSection, type HomeContent } from "@/lib/home-content";
import { localizePublicHref } from "@/lib/i18n/config";
import { localizeContent } from "@/lib/i18n/public-content";
import { getRequestLocale } from "@/lib/i18n/request";
import { metadataFromContent } from "@/lib/page-content";
import { getSiteUrl } from "@/lib/site";
import { renderHomeSection } from "./home-sections";
import { PublicMarketingLayout } from "./public-marketing-shell";
import styles from "./home.module.css";

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

      <PublicMarketingLayout
        homeHref={homeHref}
        navigation={content.navigation}
        navLabel="Primary navigation"
        footer={content.footer}
        variant="home"
      >
        <div className={styles.shell}>
          {content.sections.map(renderHomeSection)}
        </div>
      </PublicMarketingLayout>
    </main>
  );
}
