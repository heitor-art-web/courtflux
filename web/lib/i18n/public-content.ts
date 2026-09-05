import type { Locale } from "./config";
import { localizePublicHref } from "./config";
import { translateRuntimeText } from "./translations";

function localizeValue(value: unknown, locale: Locale, key?: string): unknown {
  if (typeof value === "string") {
    if (key === "href" || key === "canonical") return localizePublicHref(value, locale);
    return translateRuntimeText(locale, value);
  }
  if (Array.isArray(value)) return value.map((item) => localizeValue(item, locale));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([childKey, childValue]) => [
        childKey,
        localizeValue(childValue, locale, childKey),
      ]),
    );
  }
  return value;
}

export function localizeContent<T>(content: T, locale: Locale): T {
  if (locale === "en") return content;
  return localizeValue(content, locale) as T;
}
