import type { Metadata } from "next";
import content from "@/data/pages/video-data.json";
import { metadataFromContent } from "@/lib/page-content";
import { LegalContentPage } from "../legal-page";

export const metadata: Metadata = metadataFromContent(content.metadata);

export default function VideoDataPage() {
  return <LegalContentPage content={content} />;
}
