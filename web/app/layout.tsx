import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CourtFlux — Basketball Intelligence from Video",
  description: "CourtFlux turns basketball video into structured player, ball and event intelligence.",
  icons: {
    icon: "/brand/courtflux-app-icon.webp",
    apple: "/brand/courtflux-app-icon.webp",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
