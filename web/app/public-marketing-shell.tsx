import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";
import homeSource from "@/data/pages/home.json";
import { localeFromPathname, type Locale } from "@/lib/i18n/config";
import { localizeContent } from "@/lib/i18n/public-content";
import { LanguageSwitcher } from "./language-switcher";
import design from "./public-design-system.module.css";
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

type PublicMarketingVariant = "standard" | "home" | "legal";

function homeAnchorHref(homeHref: string, href: string) {
  if (!href.startsWith("#")) return href;
  return `${homeHref === "/" ? "/" : homeHref}${href}`;
}

function navigationModel(homeHref: string, navigation: PublicNavigationItem[]) {
  const locale = (localeFromPathname(homeHref) ?? "en") as Locale;
  const localizedHome = localizeContent(homeSource, locale);
  const globalNavigation = localizedHome.navigation.map((item) => ({
    ...item,
    href: homeAnchorHref(homeHref, item.href),
  }));

  const signIn = globalNavigation.find((item) => item.href === "/login");
  const cta = globalNavigation.find((item) => item.cta);
  const primary = globalNavigation.filter((item) => item !== signIn && item !== cta);
  const globalHrefs = new Set(globalNavigation.map((item) => item.href));

  const contextual = navigation.filter((item) => {
    if (item.cta || item.href.startsWith("/login")) return false;
    return !globalHrefs.has(homeAnchorHref(homeHref, item.href));
  });

  return { locale, primary, signIn, cta, contextual };
}

function navigationLabels(locale: Locale) {
  if (locale === "pt-PT") {
    return {
      primary: "Navegação principal",
      mobile: "Navegação principal móvel",
      menu: "Menu",
      openMenu: "Abrir navegação",
    };
  }
  if (locale === "es-ES") {
    return {
      primary: "Navegación principal",
      mobile: "Navegación principal móvil",
      menu: "Menú",
      openMenu: "Abrir navegación",
    };
  }
  return {
    primary: "Primary navigation",
    mobile: "Mobile primary navigation",
    menu: "Menu",
    openMenu: "Open navigation",
  };
}

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
  const { locale, primary, signIn, cta, contextual } = navigationModel(homeHref, navigation);
  const labels = navigationLabels(locale);

  return (
    <header className={styles.header}>
      <div className={styles.headerMain}>
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

        <nav className={styles.primaryNav} aria-label={labels.primary}>
          {primary.map((item) => (
            <Link href={item.href} key={`${item.href}-${item.label}`}>{item.label}</Link>
          ))}
        </nav>

        <div className={styles.utilities}>
          <LanguageSwitcher tone="light" />
          <details className={styles.mobileMenu}>
            <summary aria-label={labels.openMenu}>
              <span aria-hidden="true" className={styles.menuIcon}><i /><i /><i /></span>
              <span>{labels.menu}</span>
            </summary>
            <nav aria-label={labels.mobile}>
              {primary.map((item) => (
                <Link href={item.href} key={`mobile-${item.href}-${item.label}`}>{item.label}</Link>
              ))}
            </nav>
          </details>
        </div>

        <div className={styles.headerActions}>
          {signIn ? <Link className={styles.signIn} href={signIn.href}>{signIn.label}</Link> : null}
          {cta ? <Link className={styles.navCta} href={cta.href}>{cta.label}</Link> : null}
        </div>
      </div>

      {contextual.length ? (
        <nav className={styles.contextNav} aria-label={navLabel}>
          {contextual.map((item) => (
            <Link href={item.href} key={`${item.href}-${item.label}`}>{item.label}</Link>
          ))}
        </nav>
      ) : null}
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
  variant = "standard",
  children,
}: {
  homeHref: string;
  navigation: PublicNavigationItem[];
  navLabel: string;
  footer: PublicFooterContent;
  brandAlt?: string;
  variant?: PublicMarketingVariant;
  children: ReactNode;
}) {
  const variantClass = variant === "home"
    ? design.homeRoot
    : variant === "legal"
      ? design.legalRoot
      : design.standardRoot;

  return (
    <div className={`${design.publicRoot} ${variantClass}`} data-public-layout={variant}>
      <PublicHeader
        homeHref={homeHref}
        navigation={navigation}
        navLabel={navLabel}
        brandAlt={brandAlt}
      />
      {children}
      <PublicFooter homeHref={homeHref} footer={footer} brandAlt={brandAlt} />
    </div>
  );
}
