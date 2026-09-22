"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { isPublicPath } from "@/lib/i18n/config";

export function PageSurfaceBridge() {
  const pathname = usePathname();

  useEffect(() => {
    document.body.dataset.publicPage = isPublicPath(pathname) ? "true" : "false";
  }, [pathname]);

  return null;
}
