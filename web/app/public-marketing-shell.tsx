import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";
import styles from "./public-marketing-shell.module.css";

export type PublicNavigationItem = {
  label: string;
  href: string;
  cta?: boolean;
};

type PublicFooterContent = {
  meta?: string;
  links: Array<{ label: string; href: string }>;
};

export function PublicHeader({
  homeHref,
  navigation,
  navLabel,
  brandAlt = "CourtFlux — Basketball Intelligence",
}: {
  homeHref: string;
  navigation: PublicNavigationItem[];
  navLabel: string;
  brandAlt?: string;
}) {
  return (
    <header className={styles.header}>
      <Link href={homeHref} className={styles.logoLink} aria-label="CourtFlux home">
        <Image
          className={styles.logo}
          src="/brand/light_logo_safe_crop.png"
          alt={brandAlt}
          width={320}
          height={255}
          priority
        />
      </Link>
      <nav className={styles.nav} aria-label={navLabel}>
        {navigation.map((item) => (
          <Link
            className={item.cta ? styles.navCta : undefined}
            href={item.href}
            key={`${item.href}-${item.label}`}
          >
            {item.label}
          </Link>
        ))}
      </nav>
    </header>
  );
}

export function PublicFooter({
  homeHref,
  footer,
  brandAlt = "CourtFlux — Basketball Intelligence",
}: {
  homeHref: string;
  footer: PublicFooterContent;
  brandAlt?: string;
}) {
  return (
    <footer className={styles.footer}>
      <div className={styles.footerInner}>
        <div className={styles.footerBrand}>
          <Link href={homeHref} className={styles.footerLogoLink} aria-label="CourtFlux home">
            <Image
              className={styles.footerLogo}
              src="/brand/horizontal_logo_safe_crop.png"
              alt={brandAlt}
              width={660}
              height={185}
            />
          </Link>
          <span className={styles.footerMeta}>
            {footer.meta ?? "© 2026 CourtFlux · Basketball Intelligence"}
          </span>
        </div>
        <div className={styles.footerLinks}>
          {footer.links.map((item) => (
            <Link href={item.href} key={`${item.href}-${item.label}`}>{item.label}</Link>
          ))}
        </div>
      </div>
    </footer>
  );
}

export function PublicMarketingLayout({
  homeHref,
  navigation,
  navLabel,
  footer,
  brandAlt,
  children,
}: {
  homeHref: string;
  navigation: PublicNavigationItem[];
  navLabel: string;
  footer: PublicFooterContent;
  brandAlt?: string;
  children: ReactNode;
}) {
  return (
    <>
      <PublicHeader
        homeHref={homeHref}
        navigation={navigation}
        navLabel={navLabel}
        brandAlt={brandAlt}
      />
      {children}
      <PublicFooter homeHref={homeHref} footer={footer} brandAlt={brandAlt} />
    </>
  );
}
