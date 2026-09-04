import type { Metadata } from "next";
import mobileStyles from "../product-mobile.module.css";

export const metadata: Metadata = {
  title: "Set up CourtFlux",
  description: "Set up your CourtFlux basketball workspace and first team.",
  robots: {
    index: false,
    follow: false,
    nocache: true,
  },
};

export default function OnboardingLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <div className={mobileStyles.productViewport}>{children}</div>;
}
