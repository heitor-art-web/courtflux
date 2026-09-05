import type { Metadata } from "next";
import content from "@/data/pages/terms.json";
import { metadataFromContent } from "@/lib/page-content";
import { LegalContentPage } from "../legal-page";

export const metadata: Metadata = metadataFromContent(content.metadata);

export default function TermsPage() {
  return <LegalContentPage content={content} />;
}
