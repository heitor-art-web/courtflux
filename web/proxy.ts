import { NextRequest, NextResponse } from "next/server";
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

export async function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const pathLocale = localeFromPathname(pathname);
  const cleanPath = stripLocalePrefix(pathname);
  const locale: Locale = pathLocale ?? (isPublicPath(pathname) ? DEFAULT_LOCALE : cookieLocale(request));
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-courtflux-locale", locale);
  const sessionRequest = new NextRequest(request.url, { headers: requestHeaders });
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
  ],
};
