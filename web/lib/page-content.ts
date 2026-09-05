import type { Metadata } from "next";
import { localizedAlternates, localeFromPathname } from "@/lib/i18n/config";

export type EditablePageMetadata = {
  title: string;
  description: string;
  canonical: string;
  openGraph?: {
    title: string;
    description: string;
    image?: {
      url: string;
      width: number;
      height: number;
      alt: string;
    };
  };
  twitter?: {
    title: string;
    description: string;
    image?: string;
  };
};

export function metadataFromContent(content: EditablePageMetadata): Metadata {
  const locale = localeFromPathname(content.canonical) ?? "en";
  const openGraphLocale = locale === "pt-PT" ? "pt_PT" : locale === "es-ES" ? "es_ES" : "en_GB";
  const alternateLocale = ["en_GB", "pt_PT", "es_ES"].filter((item) => item !== openGraphLocale);
  const metadata: Metadata = {
    title: content.title,
    description: content.description,
    alternates: {
      canonical: content.canonical,
      languages: localizedAlternates(content.canonical),
    },
    robots: { index: true, follow: true },
  };

  if (content.openGraph) {
    metadata.openGraph = {
      type: "website",
      url: content.canonical,
      siteName: "CourtFlux",
      locale: openGraphLocale,
      alternateLocale,
      title: content.openGraph.title,
      description: content.openGraph.description,
      images: content.openGraph.image ? [content.openGraph.image] : undefined,
    };
  }

  if (content.twitter) {
    metadata.twitter = {
      card: "summary_large_image",
      title: content.twitter.title,
      description: content.twitter.description,
      images: content.twitter.image ? [content.twitter.image] : undefined,
    };
  }

  return metadata;
}
