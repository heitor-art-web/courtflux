import { NextRequest, NextResponse } from "next/server";
import { isPublicLaunchEnabled } from "@/lib/launch";
import { isP3FoundationEnabled } from "@/lib/p3/feature-flag";
import { updateSession } from "@/lib/supabase/proxy";
import {
  DEFAULT_LOCALE,
  LOCALE_COOKIE,
  isLocale,
  isPublicPath,
  localeFromPathname,
  stripLocalePrefix,
  type Locale,
} from "@/lib/i18n/config";

function cookieLocale(request: NextRequest): Locale {
  const value = request.cookies.get(LOCALE_COOKIE)?.value;
  return isLocale(value) ? value : DEFAULT_LOCALE;
}

function copyResponseCookies(from: NextResponse, to: NextResponse) {
  from.cookies.getAll().forEach((cookie) => to.cookies.set(cookie));
}

function privateBetaDeploymentResponse() {
  if (
    isPublicLaunchEnabled()
    || !process.env.VERCEL
    || process.env.VERCEL_ENV === "preview"
  ) {
    return null;
  }
  return new NextResponse("CourtFlux is currently a local private beta.", {
    status: 404,
    headers: {
      "Cache-Control": "no-store",
      "X-Robots-Tag": "noindex, nofollow, noarchive",
    },
  });
}

function p3FoundationSurface(pathname: string): "review" | "api" | null {
  if (/^\/analysis\/[^/]+\/review(?:\/|$)/.test(pathname)) return "review";
  if (
    /^\/api\/analyses\/[^/]+\/(?:things-to-review|review-actions|human-review)(?:\/|$)/.test(
      pathname,
    )
  ) {
    return "api";
  }
  return null;
}

function disabledP3Response(request: NextRequest, cleanPath: string, pathLocale: Locale | null) {
  const surface = p3FoundationSurface(cleanPath);
  if (!surface || isP3FoundationEnabled()) return null;

  if (surface === "api") {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const match = cleanPath.match(/^\/analysis\/([^/]+)\/review/);
  const runId = match?.[1];
  if (!runId) return NextResponse.redirect(new URL("/account", request.url));

  const redirectUrl = request.nextUrl.clone();
  redirectUrl.pathname = pathLocale
    ? `/${pathLocale}/analysis/${runId}`
    : `/analysis/${runId}`;
  redirectUrl.search = "";
  return NextResponse.redirect(redirectUrl);
}

export async function proxy(request: NextRequest) {
  const privateBetaResponse = privateBetaDeploymentResponse();
  if (privateBetaResponse) return privateBetaResponse;

  const { pathname } = request.nextUrl;
  const pathLocale = localeFromPathname(pathname);
  const cleanPath = stripLocalePrefix(pathname);
  const disabledResponse = disabledP3Response(request, cleanPath, pathLocale);
  if (disabledResponse) return disabledResponse;

  const publicRequest = isPublicPath(pathname);
  const locale: Locale = pathLocale ?? (publicRequest ? DEFAULT_LOCALE : cookieLocale(request));
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-courtflux-locale", locale);
  requestHeaders.set("x-courtflux-public", publicRequest ? "1" : "0");
  const sessionRequest = new NextRequest(request, { headers: requestHeaders });
  const sessionResponse = await updateSession(sessionRequest);

  if (sessionResponse.status >= 300 && sessionResponse.status < 400) return sessionResponse;

  if (pathLocale) {
    const rewriteUrl = request.nextUrl.clone();
    rewriteUrl.pathname = cleanPath;
    const response = NextResponse.rewrite(rewriteUrl, { request: { headers: requestHeaders } });
    copyResponseCookies(sessionResponse, response);
    response.cookies.set(LOCALE_COOKIE, locale, {
      path: "/",
      maxAge: 60 * 60 * 24 * 365,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
    });
    return response;
  }

  return sessionResponse;
}

export const config = {
  matcher: [
    "/((?!api|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp|ico|webmanifest)$).*)",
    "/api/analyses/:id/things-to-review",
    "/api/analyses/:id/review-actions",
    "/api/analyses/:id/human-review",
  ],
};
