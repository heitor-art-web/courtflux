"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { isPublicPath, type Locale } from "@/lib/i18n/config";
import { translateRuntimeText } from "@/lib/i18n/translations";

const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "CODE", "PRE", "NOSCRIPT", "SVG"]);
const TRANSLATED_ATTRIBUTES = ["placeholder", "aria-label", "title"] as const;
const MONTHS: Record<string, number> = {
  Jan: 0,
  Feb: 1,
  Mar: 2,
  Apr: 3,
  May: 4,
  Jun: 5,
  Jul: 6,
  Aug: 7,
  Sep: 8,
  Oct: 9,
  Nov: 10,
  Dec: 11,
};
const EN_DATE_PATTERN = /\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})(?:,\s+(\d{2}):(\d{2}))?\b/g;

function ignored(element: Element | null) {
  if (!element) return false;
  return Boolean(element.closest("[data-i18n-ignore]"));
}

function localizeEnglishDates(value: string, locale: Locale) {
  if (locale === "en") return value;
  return value.replace(EN_DATE_PATTERN, (_match, dayText: string, monthText: string, yearText: string, hourText?: string, minuteText?: string) => {
    const day = Number(dayText);
    const year = Number(yearText);
    const month = MONTHS[monthText];
    const hour = hourText ? Number(hourText) : 12;
    const minute = minuteText ? Number(minuteText) : 0;
    if (!Number.isInteger(day) || !Number.isInteger(year) || month === undefined) return _match;

    const date = new Date(year, month, day, hour, minute);
    if (Number.isNaN(date.getTime())) return _match;

    return new Intl.DateTimeFormat(locale, {
      day: "numeric",
      month: "short",
      year: "numeric",
      ...(hourText ? { hour: "2-digit", minute: "2-digit" } : {}),
    }).format(date);
  });
}

function translatedValue(locale: Locale, value: string) {
  return translateRuntimeText(locale, localizeEnglishDates(value, locale));
}

function translateElementAttributes(element: Element, locale: Locale) {
  if (ignored(element) || SKIP_TAGS.has(element.tagName)) return;
  for (const attribute of TRANSLATED_ATTRIBUTES) {
    const value = element.getAttribute(attribute);
    if (!value) continue;
    const translated = translatedValue(locale, value);
    if (translated !== value) element.setAttribute(attribute, translated);
  }
}

function translateTree(root: Node, locale: Locale) {
  if (locale === "en") return;

  if (root instanceof Element) translateElementAttributes(root, locale);

  const documentRoot = root instanceof Document ? root.documentElement : root;
  const walker = document.createTreeWalker(documentRoot, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  let current: Node | null = walker.currentNode;

  while (current) {
    if (current.nodeType === Node.ELEMENT_NODE) {
      translateElementAttributes(current as Element, locale);
    } else if (current.nodeType === Node.TEXT_NODE) {
      const parent = current.parentElement;
      if (parent && !SKIP_TAGS.has(parent.tagName) && !ignored(parent)) {
        const value = current.nodeValue;
        if (value) {
          const translated = translatedValue(locale, value);
          if (translated !== value) current.nodeValue = translated;
        }
      }
    }
    current = walker.nextNode();
  }
}

export function LocaleTextBridge({ locale }: { locale: Locale }) {
  const pathname = usePathname();

  useEffect(() => {
    if (locale === "en" || isPublicPath(pathname)) return;
    translateTree(document, locale);

    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === "characterData") {
          translateTree(mutation.target, locale);
          continue;
        }
        mutation.addedNodes.forEach((node) => translateTree(node, locale));
      }
    });

    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, [locale, pathname]);

  return null;
}
