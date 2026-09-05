import type { Metadata } from "next";
import englishContent from "@/data/pages/teams.json";
import { AudienceLandingPage } from "../audience-landing-page";
import { metadataFromContent } from "@/lib/page-content";
import { getRequestLocale } from "@/lib/i18n/request";
import { localizeContent } from "@/lib/i18n/public-content";

export async function generateMetadata(): Promise<Metadata> {
  const locale = await getRequestLocale();
  return metadataFromContent(localizeContent(englishContent, locale).metadata);
}

export default async function TeamsLandingPage() {
  const locale = await getRequestLocale();
  const content = localizeContent(englishContent, locale);
  return <AudienceLandingPage content={content} locale={locale} />;
}
