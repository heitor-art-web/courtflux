import type { MetadataRoute } from "next";
import { isPublicLaunchEnabled } from "@/lib/launch";
import { getSiteUrl } from "@/lib/site";

export default function robots(): MetadataRoute.Robots {
  if (!isPublicLaunchEnabled()) {
    return {
      rules: [{ userAgent: "*", disallow: "/" }],
    };
  }

  const siteUrl = getSiteUrl();

  return {
    rules: [
      {
        userAgent: "*",
        allow: "/",
        disallow: ["/api/", "/account", "/login", "/lab"],
      },
    ],
    sitemap: `${siteUrl}/sitemap.xml`,
    host: siteUrl,
  };
}
