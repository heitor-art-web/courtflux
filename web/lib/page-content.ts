import type { Metadata } from "next";

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
  const metadata: Metadata = {
    title: content.title,
    description: content.description,
    alternates: { canonical: content.canonical },
    robots: { index: true, follow: true },
  };

  if (content.openGraph) {
    metadata.openGraph = {
      type: "website",
      url: content.canonical,
      siteName: "CourtFlux",
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
