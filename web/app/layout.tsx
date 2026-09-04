import type { Metadata, Viewport } from "next";
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
    "CourtFlux is basketball video analysis software for coaches, teams and academies, turning ordinary game film into structured player, ball and spatial intelligence.",
  authors: [{ name: "CourtFlux" }],
  creator: "CourtFlux",
  publisher: "CourtFlux",
  category: "Sports technology",
  referrer: "origin-when-cross-origin",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  icons: {
    icon: [
      { url: "/brand/courtflux_app_icon_32px.png", sizes: "32x32", type: "image/png" },
      { url: "/brand/courtflux_app_icon_64px.png", sizes: "64x64", type: "image/png" },
    ],
    apple: "/brand/courtflux_app_icon_256px.png",
  },
};

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: "#06111f",
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
