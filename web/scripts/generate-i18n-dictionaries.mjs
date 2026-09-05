import { readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(here, "..");
const locales = ["pt-PT", "es-ES"];
const propertyPattern = /^\s*("(?:\\.|[^"\\])*")\s*:/;

function dedupeDictionary(source, locale) {
  const lines = source.split("\n");
  const lastIndexByKey = new Map();

  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index].match(propertyPattern);
    if (!match) continue;
    const key = JSON.parse(match[1]);
    lastIndexByKey.set(key, index);
  }

  const output = lines.filter((line, index) => {
    const match = line.match(propertyPattern);
    if (!match) return true;
    const key = JSON.parse(match[1]);
    return lastIndexByKey.get(key) === index;
  });

  const removed = lines.length - output.length;
  if (removed === 0) {
    console.log(`[i18n] ${locale}: no duplicate keys found`);
  } else {
    console.log(`[i18n] ${locale}: removed ${removed} duplicate key${removed === 1 ? "" : "s"}`);
  }

  return output.join("\n");
}

for (const locale of locales) {
  const sourcePath = resolve(webRoot, "lib", "i18n", "dictionaries", `${locale}.txt`);
  const outputPath = resolve(webRoot, "lib", "i18n", `${locale}.ts`);
  const source = await readFile(sourcePath, "utf8");
  await writeFile(outputPath, dedupeDictionary(source, locale), "utf8");
}
