import type { Metadata } from "next";
import mobileStyles from "../product-mobile.module.css";

export const metadata: Metadata = {
  title: "Account",
  robots: {
    index: false,
    follow: false,
    nocache: true,
  },
};

export default function AccountLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <div className={mobileStyles.productViewport}>{children}</div>;
}
