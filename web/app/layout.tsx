import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Basketball AI Analytics — Beta",
  description: "Results-first basketball video analysis beta.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
