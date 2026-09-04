import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "New Analysis",
  description: "Start a private CourtFlux basketball video analysis.",
  robots: {
    index: false,
    follow: false,
    nocache: true,
  },
};

export default function NewAnalysisLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
