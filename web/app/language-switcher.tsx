"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { usePathname, useRouter } from "next/navigation";
import {
  LOCALE_COOKIE,
  LOCALE_LABELS,
  SUPPORTED_LOCALES,
  isPublicPath,
  localeFromPathname,
  localizePublicHref,
  stripLocalePrefix,
  type Locale,
} from "@/lib/i18n/config";
import styles from "./language-switcher.module.css";

type LanguageSwitcherTone = "dark" | "light";

function destinationPath(pathname: string, locale: Locale) {
  const cleanPath = stripLocalePrefix(pathname);
  return isPublicPath(pathname) ? localizePublicHref(cleanPath, locale) : cleanPath;
}

export function LanguageSwitcher({
  initialLocale,
  tone = "dark",
}: {
  initialLocale?: Locale;
  tone?: LanguageSwitcherTone;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const routeLocale = localeFromPathname(pathname) ?? "en";
  const [locale, setLocale] = useState<Locale>(initialLocale ?? routeLocale);

  useEffect(() => {
    setLocale(initialLocale ?? localeFromPathname(pathname) ?? "en");
  }, [initialLocale, pathname]);

  function changeLanguage(nextLocale: Locale) {
    if (nextLocale === locale) return;

    setLocale(nextLocale);
    document.cookie = `${LOCALE_COOKIE}=${encodeURIComponent(nextLocale)}; Max-Age=31536000; Path=/; SameSite=Lax`;

    const nextPath = destinationPath(pathname, nextLocale);
    const destination = `${nextPath}${window.location.search}${window.location.hash}`;

    if (isPublicPath(pathname)) {
      router.push(destination);
      return;
    }

    if (nextPath !== pathname) router.push(destination);
    router.refresh();
  }

  const label = locale === "en" ? "Language" : locale === "pt-PT" ? "Língua" : "Idioma";
  const toneClass = tone === "light" ? styles.light : "";

  return (
    <label className={`${styles.switcher} ${toneClass}`} data-i18n-ignore>
      <span>{label}</span>
      <select
        aria-label={label}
        value={locale}
        onChange={(event) => changeLanguage(event.target.value as Locale)}
      >
        {SUPPORTED_LOCALES.map((option) => (
          <option key={option} value={option}>{LOCALE_LABELS[option]}</option>
        ))}
      </select>
    </label>
  );
}

export function AppLanguageSwitcherBridge({ initialLocale }: { initialLocale: Locale }) {
  const pathname = usePathname();
  const [host, setHost] = useState<HTMLElement | null>(null);

  useEffect(() => {
    if (isPublicPath(pathname)) {
      setHost(null);
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      setHost(document.querySelector<HTMLElement>("#main-content header"));
    });

    return () => window.cancelAnimationFrame(frame);
  }, [pathname]);

  if (isPublicPath(pathname) || !host) return null;
  return createPortal(<LanguageSwitcher initialLocale={initialLocale} />, host);
}
