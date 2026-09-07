import type { Metadata, Viewport } from "next";
import "./globals.css";
import "./accessibility.css";
import "./homepage-light.css";
import "./footer-brand-fix.css";
import { getSiteUrl } from "@/lib/site";
import { getRequestLocale } from "@/lib/i18n/request";
import { LanguageSwitcher } from "./language-switcher";
import { LocaleTextBridge } from "./locale-text-bridge";
import { Analytics } from "@vercel/analytics/next";

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
  colorScheme: "light dark",
  themeColor: "#ffffff",
};

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const locale = await getRequestLocale();
  const skipLabel = locale === "pt-PT"
    ? "Saltar para o conteúdo principal"
    : locale === "es-ES"
      ? "Saltar al contenido principal"
      : "Skip to main content";

  return (
    <html lang={locale}>
      <body>
        <a className="skip-link" href="#main-content">{skipLabel}</a>
        <div id="main-content" tabIndex={-1}>
          {children}
        </div>
        <LocaleTextBridge locale={locale} />
        <LanguageSwitcher initialLocale={locale} />
        <Analytics />
      </body>
    </html>
  );
}
