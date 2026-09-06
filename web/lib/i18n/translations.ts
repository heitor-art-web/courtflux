import type { Locale } from "./config";
import { APP_CORE_TRANSLATIONS } from "./app-core";
import ptPT from "./pt-PT";
import ptPTPolish from "./pt-PT-polish";
import esES from "./es-ES";
import esESPolish from "./es-ES-polish";

const DICTIONARIES: Record<Exclude<Locale, "en">, Record<string, string>> = {
  "pt-PT": { ...ptPT, ...ptPTPolish, ...APP_CORE_TRANSLATIONS["pt-PT"] },
  "es-ES": { ...esES, ...esESPolish, ...APP_CORE_TRANSLATIONS["es-ES"] },
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
        [/^Shorten the analysis window to fit the (.+) min remaining this month\.$/, (m) => `Reduza o intervalo de análise para caber nos ${m[1]} min restantes deste mês.`],
        [/^(\d+(?:[.,]\d+)?) min remaining$/, (m) => `${m[1]} min restantes`],
        [/^remaining from (.+) min this month$/, (m) => `restantes de ${m[1]} min este mês`],
        [/^of (.+) analysis minutes$/, (m) => `de ${m[1]} minutos de análise`],
        [/^(\d+)% of monthly analysis allowance used$/, (m) => `${m[1]}% do limite mensal de análise utilizado`],
        [/^(\d+)% of this month's allowance used\.$/, (m) => `${m[1]}% do limite deste mês utilizado.`],
        [/^(\d+) saved roster entries available$/, (m) => `${m[1]} jogadores guardados no plantel`],
        [/^(.+) plan$/, (m) => `Plano ${m[1]}`],
        [/^(.+) roster$/, (m) => `Plantel — ${m[1]}`],
        [/^Until (.+)$/, (m) => `Até ${m[1]}`],
        [/^(.+) includes (.+) analysis minutes per month\.$/, (m) => `${m[1]} inclui ${m[2]} minutos de análise por mês.`],
        [/^This workspace currently has (.+) additional beta minutes\.$/, (m) => `Este espaço de trabalho tem atualmente ${m[1]} minutos beta adicionais.`],
      ]
    : [
        [/^Start analysis · (.+)$/, (m) => `Iniciar análisis · ${m[1]}`],
        [/^Shorten the analysis window to fit the (.+) min remaining this month\.$/, (m) => `Reduce el intervalo de análisis para ajustarlo a los ${m[1]} min restantes de este mes.`],
        [/^(\d+(?:[.,]\d+)?) min remaining$/, (m) => `${m[1]} min restantes`],
        [/^remaining from (.+) min this month$/, (m) => `restantes de ${m[1]} min este mes`],
        [/^of (.+) analysis minutes$/, (m) => `de ${m[1]} minutos de análisis`],
        [/^(\d+)% of monthly analysis allowance used$/, (m) => `${m[1]}% del límite mensual de análisis utilizado`],
        [/^(\d+)% of this month's allowance used\.$/, (m) => `${m[1]}% del límite de este mes utilizado.`],
        [/^(\d+) saved roster entries available$/, (m) => `${m[1]} jugadores guardados en la plantilla`],
        [/^(.+) plan$/, (m) => `Plan ${m[1]}`],
        [/^(.+) roster$/, (m) => `Plantilla — ${m[1]}`],
        [/^Until (.+)$/, (m) => `Hasta ${m[1]}`],
        [/^(.+) includes (.+) analysis minutes per month\.$/, (m) => `${m[1]} incluye ${m[2]} minutos de análisis al mes.`],
        [/^This workspace currently has (.+) additional beta minutes\.$/, (m) => `Este espacio de trabajo tiene actualmente ${m[1]} minutos beta adicionales.`],
      ];

  for (const [pattern, replacement] of dynamic) {
    const match = core.match(pattern);
    if (match) return `${leading}${replacement(match)}${trailing}`;
  }
  return value;
}
