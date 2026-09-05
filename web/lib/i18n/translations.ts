import type { Locale } from "./config";
import ptPT from "./pt-PT";
import esES from "./es-ES";

const DICTIONARIES: Record<Exclude<Locale, "en">, Record<string, string>> = {
  "pt-PT": ptPT,
  "es-ES": esES,
};

export function translateText(locale: Locale, value: string): string {
  if (locale === "en" || !value) return value;
  return DICTIONARIES[locale][value] ?? value;
}

export function translateRuntimeText(locale: Locale, value: string): string {
  if (locale === "en" || !value.trim()) return value;
  const leading = value.match(/^\s*/)?.[0] ?? "";
  const trailing = value.match(/\s*$/)?.[0] ?? "";
  const core = value.trim();
  const direct = translateText(locale, core);
  if (direct !== core) return `${leading}${direct}${trailing}`;

  const dynamic: Array<[RegExp, (match: RegExpMatchArray) => string]> = locale === "pt-PT"
    ? [
        [/^Start analysis · (.+)$/, (m) => `Iniciar análise · ${m[1]}`],
        [/^(\d+(?:[.,]\d+)?) min remaining$/, (m) => `${m[1]} min restantes`],
        [/^of (.+) analysis minutes$/, (m) => `de ${m[1]} minutos de análise`],
        [/^(\d+)% of monthly analysis allowance used$/, (m) => `${m[1]}% do limite mensal de análise utilizado`],
        [/^(\d+) saved roster entries available$/, (m) => `${m[1]} jogadores guardados no plantel`],
      ]
    : [
        [/^Start analysis · (.+)$/, (m) => `Iniciar análisis · ${m[1]}`],
        [/^(\d+(?:[.,]\d+)?) min remaining$/, (m) => `${m[1]} min restantes`],
        [/^of (.+) analysis minutes$/, (m) => `de ${m[1]} minutos de análisis`],
        [/^(\d+)% of monthly analysis allowance used$/, (m) => `${m[1]}% del límite mensual de análisis utilizado`],
        [/^(\d+) saved roster entries available$/, (m) => `${m[1]} jugadores guardados en la plantilla`],
      ];

  for (const [pattern, replacement] of dynamic) {
    const match = core.match(pattern);
    if (match) return `${leading}${replacement(match)}${trailing}`;
  }
  return value;
}
