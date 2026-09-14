import type { Metadata } from "next";
import { notFound } from "next/navigation";

export const metadata: Metadata = {
  title: "CourtFlux Beta Lab",
  robots: {
    index: false,
    follow: false,
    nocache: true,
  },
};

export default function LabLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  if (process.env.NODE_ENV === "production") {
    notFound();
  }

  return children;
}
