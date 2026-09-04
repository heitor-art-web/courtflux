import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "CourtFlux Beta Lab",
  robots: {
    index: false,
    follow: false,
    nocache: true,
  },
};

export default function LabLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return children;
}
