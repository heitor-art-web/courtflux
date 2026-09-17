import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";
import shared from "@/data/pages/shared.json";
import { localizePublicHref, type Locale } from "@/lib/i18n/config";
import { localizeContent } from "@/lib/i18n/public-content";
import styles from "./legal.module.css";

type RichTextPart = {
  text: string;
  href?: string;
  strong?: boolean;
};

type LegalSectionContent = {
  title: string;
  paragraphs?: RichTextPart[][];
  bullets?: string[];
  retention?: Array<{ label: string; value: string }>;
};

export type LegalPageContent = {
  eyebrow: string;
  title: string;
  intro: string;
  notice?: RichTextPart[];
  sections: LegalSectionContent[];
};

function RichText({ parts }: { parts: RichTextPart[] }) {
  return (
    <>
      {parts.map((part, index) => {
        let node: ReactNode = part.text;
        if (part.strong) node = <strong>{node}</strong>;
        if (part.href) node = <Link href={part.href}>{node}</Link>;
        return <span key={`${part.text}-${index}`}>{node}</span>;
      })}
    </>
  );
}

export function LegalPage({
  eyebrow,
  title,
  intro,
  notice,
  children,
  locale = "en",
}: {
  eyebrow: string;
  title: string;
  intro: string;
  notice?: ReactNode;
  children: ReactNode;
  locale?: Locale;
}) {
  const localizedShared = localizeContent(shared, locale);
  const homeHref = localizePublicHref("/", locale);

  return (
    <main className={styles.legalPage}>
      <header className={styles.header}>
        <Link className={styles.brand} href={homeHref} aria-label="CourtFlux home">
          <Image
            src="/brand/light_logo_safe_crop.png"
            alt={localizedShared.brandAlt}
            width={320}
            height={255}
            priority
          />
        </Link>
        <nav className={styles.nav} aria-label="Trust and legal navigation">
          {localizedShared.legalNavigation.map((item) => (
            <Link href={item.href} key={item.href}>{item.label}</Link>
          ))}
        </nav>
      </header>

      <div className={styles.content}>
        <section className={styles.hero}>
          <div className={styles.eyebrow}>{eyebrow}</div>
          <h1>{title}</h1>
          <p>{intro}</p>
          {notice ? <div className={styles.notice}>{notice}</div> : null}
        </section>
        <div className={styles.sections}>{children}</div>
      </div>

      <footer className={styles.footer}>
        <span>{localizedShared.legalFooter.copyright}</span>
        <div className={styles.footerLinks}>
          {localizedShared.legalFooter.links.map((item) => (
            <Link href={item.href} key={item.href}>{item.label}</Link>
          ))}
        </div>
      </footer>
    </main>
  );
}

export function LegalContentPage({ content, locale = "en" }: { content: LegalPageContent; locale?: Locale }) {
  return (
    <LegalPage
      eyebrow={content.eyebrow}
      title={content.title}
      intro={content.intro}
      notice={content.notice ? <RichText parts={content.notice} /> : undefined}
      locale={locale}
    >
      {content.sections.map((section) => (
        <section className={styles.section} key={section.title}>
          <h2>{section.title}</h2>
          {section.retention ? (
            <div className={styles.retention}>
              {section.retention.map((row) => (
                <div key={row.label}><span>{row.label}</span><strong>{row.value}</strong></div>
              ))}
            </div>
          ) : null}
          {section.bullets ? (
            <ul>{section.bullets.map((item) => <li key={item}>{item}</li>)}</ul>
          ) : null}
          {section.paragraphs?.map((parts, index) => (
            <p key={`${section.title}-${index}`}><RichText parts={parts} /></p>
          ))}
        </section>
      ))}
    </LegalPage>
  );
}

export { styles as legalStyles };
