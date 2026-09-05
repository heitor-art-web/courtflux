import type { Metadata } from "next";
import englishContent from "@/data/pages/video-data.json";
import { metadataFromContent } from "@/lib/page-content";
import { getRequestLocale } from "@/lib/i18n/request";
import { localizeContent } from "@/lib/i18n/public-content";
import { LegalContentPage } from "../legal-page";

export async function generateMetadata(): Promise<Metadata> {
  const locale = await getRequestLocale();
  return metadataFromContent(localizeContent(englishContent, locale).metadata);
}

export default async function VideoDataPage() {
  const locale = await getRequestLocale();
  const content = localizeContent(englishContent, locale);
  return <LegalContentPage content={content} locale={locale} />;
}
