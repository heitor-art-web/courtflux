"use client";

import { useEffect } from "react";
import type { Locale } from "@/lib/i18n/config";
import { translateRuntimeText } from "@/lib/i18n/translations";

const SKIP_TAGS = new Set(["SCRIPT", "STYLE", "CODE", "PRE", "NOSCRIPT", "SVG"]);
const TRANSLATED_ATTRIBUTES = ["placeholder", "aria-label", "title"] as const;

function ignored(element: Element | null) {
  if (!element) return false;
  return Boolean(element.closest("[data-i18n-ignore]"));
}

function translateElementAttributes(element: Element, locale: Locale) {
  if (ignored(element) || SKIP_TAGS.has(element.tagName)) return;
  for (const attribute of TRANSLATED_ATTRIBUTES) {
    const value = element.getAttribute(attribute);
    if (!value) continue;
    const translated = translateRuntimeText(locale, value);
    if (translated !== value) element.setAttribute(attribute, translated);
  }
}

function translateTree(root: Node, locale: Locale) {
  if (locale === "en") return;

  if (root instanceof Element) translateElementAttributes(root, locale);

  const documentRoot = root instanceof Document ? root.documentElement : root;
  const walker = document.createTreeWalker(documentRoot, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  let current = walker.currentNode;

  while (current) {
    if (current.nodeType === Node.ELEMENT_NODE) {
      translateElementAttributes(current as Element, locale);
    } else if (current.nodeType === Node.TEXT_NODE) {
      const parent = current.parentElement;
      if (parent && !SKIP_TAGS.has(parent.tagName) && !ignored(parent)) {
        const value = current.nodeValue;
        if (value) {
          const translated = translateRuntimeText(locale, value);
          if (translated !== value) current.nodeValue = translated;
        }
      }
    }
    current = walker.nextNode();
  }
}

export function LocaleTextBridge({ locale }: { locale: Locale }) {
  useEffect(() => {
    if (locale === "en") return;
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
  }, [locale]);

  return null;
}
