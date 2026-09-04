import { createBrowserClient } from "@supabase/ssr";

function requireBrowserEnv(name: string, value: string | undefined) {
  if (!value) {
    throw new Error(`${name} is not configured`);
  }
  return value;
}

export function createClient() {
  return createBrowserClient(
    requireBrowserEnv("NEXT_PUBLIC_SUPABASE_URL", process.env.NEXT_PUBLIC_SUPABASE_URL),
    requireBrowserEnv(
      "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY",
      process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY,
    ),
  );
}
