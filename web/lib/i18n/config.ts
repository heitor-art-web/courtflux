export const SUPPORTED_LOCALES = ["en", "pt-PT", "es-ES"] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];

export const DEFAULT_LOCALE: Locale = "en";
export const LOCALE_COOKIE = "courtflux_locale";

export const LOCALE_SLUGS: Record<Locale, string> = {
  en: "",
  "pt-PT": "pt-pt",
  "es-ES": "es-es",
};

export const LOCALE_LABELS: Record<Locale, string> = {
  en: "English",
  "pt-PT": "Português (Portugal)",
  "es-ES": "Español (España)",
};

export const PUBLIC_PATHS = [
  "/",
  "/basketball-video-analysis",
  "/basketball-analysis-for-coaches",
  "/basketball-analysis-for-teams",
  "/trust",
  "/video-data",
  "/privacy",
  "/terms",
  "/ai-limitations",
] as const;

const PUBLIC_PATH_SET = new Set<string>(PUBLIC_PATHS);

export function isLocale(value: string | null | undefined): value is Locale {
  return SUPPORTED_LOCALES.includes(value as Locale);
}

export function localeFromSlug(value: string | null | undefined): Locale | null {
  if (!value) return null;
  const slug = value.toLowerCase();
  if (slug === "pt-pt") return "pt-PT";
  if (slug === "es-es") return "es-ES";
  if (slug === "en") return "en";
  return null;
}

export function localePrefix(locale: Locale): string {
  const slug = LOCALE_SLUGS[locale];
  return slug ? `/${slug}` : "";
}

export function stripLocalePrefix(pathname: string): string {
  const match = pathname.match(/^\/(pt-pt|es-es)(?=\/|$)/i);
  if (!match) return pathname || "/";
  const stripped = pathname.slice(match[0].length);
  return stripped || "/";
}

export function localeFromPathname(pathname: string): Locale | null {
  const match = pathname.match(/^\/(pt-pt|es-es)(?=\/|$)/i);
  return localeFromSlug(match?.[1]);
}

export function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATH_SET.has(stripLocalePrefix(pathname));
}

function splitHref(href: string) {
  const hashIndex = href.indexOf("#");
  const beforeHash = hashIndex >= 0 ? href.slice(0, hashIndex) : href;
  const hash = hashIndex >= 0 ? href.slice(hashIndex) : "";
  const queryIndex = beforeHash.indexOf("?");
  const pathname = queryIndex >= 0 ? beforeHash.slice(0, queryIndex) : beforeHash;
  const query = queryIndex >= 0 ? beforeHash.slice(queryIndex) : "";
  return { pathname, query, hash };
}

export function localizePublicHref(href: string, locale: Locale): string {
  if (!href || href.startsWith("#") || /^(?:https?:|mailto:|tel:)/i.test(href)) return href;
  const { pathname, query, hash } = splitHref(href);
  if (!pathname.startsWith("/")) return href;
  const cleanPath = stripLocalePrefix(pathname || "/");
  if (!PUBLIC_PATH_SET.has(cleanPath)) return `${cleanPath}${query}${hash}`;
  if (locale === "en") return `${cleanPath}${query}${hash}`;
  const prefix = localePrefix(locale);
  const localizedPath = cleanPath === "/" ? prefix || "/" : `${prefix}${cleanPath}`;
  return `${localizedPath}${query}${hash}`;
}

export function localizedAlternates(pathname: string): Record<string, string> {
  const cleanPath = stripLocalePrefix(pathname || "/");
  return {
    en: localizePublicHref(cleanPath, "en"),
    "pt-PT": localizePublicHref(cleanPath, "pt-PT"),
    "es-ES": localizePublicHref(cleanPath, "es-ES"),
    "x-default": localizePublicHref(cleanPath, "en"),
  };
}
