import { AuthForm } from "./auth-form";

type LoginSearchParams = Promise<{
  mode?: string | string[];
}>;

export default async function LoginPage({ searchParams }: { searchParams: LoginSearchParams }) {
  const params = await searchParams;
  const requestedMode = Array.isArray(params.mode) ? params.mode[0] : params.mode;
  const initialMode = requestedMode === "signup" ? "signup" : "login";

  return <AuthForm initialMode={initialMode} />;
}
