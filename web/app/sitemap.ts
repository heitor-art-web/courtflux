import type { MetadataRoute } from "next";
import { localizePublicHref, SUPPORTED_LOCALES } from "@/lib/i18n/config";
import { isPublicLaunchEnabled } from "@/lib/launch";
import { getSiteUrl } from "@/lib/site";

const pages = [
  { path: "/", changeFrequency: "weekly" as const, priority: 1 },
  { path: "/basketball-video-analysis", changeFrequency: "weekly" as const, priority: 0.9 },
  { path: "/basketball-analysis-for-coaches", changeFrequency: "weekly" as const, priority: 0.85 },
  { path: "/basketball-analysis-for-teams", changeFrequency: "weekly" as const, priority: 0.85 },
  { path: "/trust", changeFrequency: "monthly" as const, priority: 0.5 },
  { path: "/video-data", changeFrequency: "monthly" as const, priority: 0.4 },
  { path: "/ai-limitations", changeFrequency: "monthly" as const, priority: 0.4 },
  { path: "/privacy", changeFrequency: "monthly" as const, priority: 0.3 },
  { path: "/terms", changeFrequency: "monthly" as const, priority: 0.3 },
];

export default function sitemap(): MetadataRoute.Sitemap {
  if (!isPublicLaunchEnabled()) return [];

  const siteUrl = getSiteUrl();
  const lastModified = new Date();

  return pages.flatMap((page) => {
    const languages = {
      en: `${siteUrl}${localizePublicHref(page.path, "en") === "/" ? "" : localizePublicHref(page.path, "en")}`,
      "pt-PT": `${siteUrl}${localizePublicHref(page.path, "pt-PT")}`,
      "es-ES": `${siteUrl}${localizePublicHref(page.path, "es-ES")}`,
      "x-default": `${siteUrl}${localizePublicHref(page.path, "en") === "/" ? "" : localizePublicHref(page.path, "en")}`,
    };

    return SUPPORTED_LOCALES.map((locale) => {
      const href = localizePublicHref(page.path, locale);
      return {
        url: `${siteUrl}${href === "/" ? "" : href}`,
        lastModified,
        changeFrequency: page.changeFrequency,
        priority: page.priority,
        alternates: { languages },
      };
    });
  });
}
