import type { Metadata } from "next";
import content from "@/data/pages/teams.json";
import { AudienceLandingPage } from "../audience-landing-page";
import { metadataFromContent } from "@/lib/page-content";

export const metadata: Metadata = metadataFromContent(content.metadata);

export default function TeamsLandingPage() {
  return <AudienceLandingPage content={content} />;
}
