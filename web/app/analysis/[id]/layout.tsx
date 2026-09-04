import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Analysis | CourtFlux",
  robots: {
    index: false,
    follow: false,
    nocache: true,
  },
};

export default function AnalysisLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
