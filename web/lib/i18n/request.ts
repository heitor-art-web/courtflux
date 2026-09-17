import { headers } from "next/headers";
import { DEFAULT_LOCALE, isLocale, type Locale } from "./config";

export async function getRequestLocale(): Promise<Locale> {
  const requestHeaders = await headers();
  const locale = requestHeaders.get("x-courtflux-locale");
  return isLocale(locale) ? locale : DEFAULT_LOCALE;
}

export async function isPublicRequest(): Promise<boolean> {
  const requestHeaders = await headers();
  return requestHeaders.get("x-courtflux-public") === "1";
}
