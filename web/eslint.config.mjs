import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    files: ["app/lab/page.tsx"],
    rules: {
      // The Lab performs a one-time health request to synchronize with the external worker.
      // Keeping this exception local avoids weakening the rule across customer-facing UI.
      "react-hooks/set-state-in-effect": "off",
    },
  },
  globalIgnores([
    ".next/**",
    "node_modules/**",
    "next-env.d.ts",
  ]),
]);
