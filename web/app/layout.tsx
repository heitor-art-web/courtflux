import type { Metadata } from "next";
import "./globals.css";
import "./accessibility.css";
import { getSiteUrl } from "@/lib/site";

const siteUrl = getSiteUrl();

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  applicationName: "CourtFlux",
  manifest: "/manifest.webmanifest",
  title: {
    default: "CourtFlux — Basketball Video Analysis & Intelligence",
    template: "%s | CourtFlux",
  },
  description:
    "CourtFlux turns basketball game film into structured player, ball, spatial and coaching intelligence for coaches, teams and academies.",
  authors: [{ name: "CourtFlux" }],
  creator: "CourtFlux",
  publisher: "CourtFlux",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  icons: {
    icon: [
      { url: "/brand/courtflux_app_icon_32px.png", sizes: "32x32", type: "image/png" },
      { url: "/brand/courtflux_app_icon_64px.png", sizes: "64x64", type: "image/png" },
    ],
    apple: "/brand/courtflux_app_icon_256px.png",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">Skip to main content</a>
        <div id="main-content" tabIndex={-1}>
          {children}
        </div>
      </body>
    </html>
  );
}
