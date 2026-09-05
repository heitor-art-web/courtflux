"use client";

import { useEffect, useState } from "react";
import type { Locale } from "@/lib/i18n/config";
import styles from "./language-switcher.module.css";

const COOKIE_NAME = "courtflux_locale";
const PUBLIC_PATHS = new Set([
  "/",
  "/basketball-video-analysis",
  "/basketball-analysis-for-coaches",
  "/basketball-analysis-for-teams",
  "/trust",
  "/video-data",
  "/privacy",
  "/terms",
  "/ai-limitations",
]);

const OPTIONS: Array<{ locale: Locale; label: string }> = [
  { locale: "en", label: "English" },
  { locale: "pt-PT", label: "Português (Portugal)" },
  { locale: "es-ES", label: "Español (España)" },
];

function stripLocale(pathname: string) {
  const stripped = pathname.replace(/^\/(pt-pt|es-es)(?=\/|$)/i, "");
  return stripped || "/";
}

function targetPath(pathname: string, locale: Locale) {
  const cleanPath = stripLocale(pathname);
  if (!PUBLIC_PATHS.has(cleanPath)) return cleanPath;
  if (locale === "pt-PT") return cleanPath === "/" ? "/pt-pt" : `/pt-pt${cleanPath}`;
  if (locale === "es-ES") return cleanPath === "/" ? "/es-es" : `/es-es${cleanPath}`;
  return cleanPath;
}

export function LanguageSwitcher({ initialLocale }: { initialLocale: Locale }) {
  const [locale, setLocale] = useState(initialLocale);

  useEffect(() => setLocale(initialLocale), [initialLocale]);

  function changeLanguage(nextLocale: Locale) {
    setLocale(nextLocale);
    document.cookie = `${COOKIE_NAME}=${encodeURIComponent(nextLocale)}; Max-Age=31536000; Path=/; SameSite=Lax`;
    const destination = `${targetPath(window.location.pathname, nextLocale)}${window.location.search}${window.location.hash}`;
    window.location.assign(destination);
  }

  const label = locale === "pt-PT" ? "Idioma" : locale === "es-ES" ? "Idioma" : "Language";

  return (
    <label className={styles.switcher} data-i18n-ignore>
      <span>{label}</span>
      <select
        aria-label={label}
        value={locale}
        onChange={(event) => changeLanguage(event.target.value as Locale)}
      >
        {OPTIONS.map((option) => (
          <option key={option.locale} value={option.locale}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}
