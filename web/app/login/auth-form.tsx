"use client";

import { FormEvent, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { getAuthConfirmationUrl } from "@/lib/auth-url";
import { createClient } from "@/lib/supabase/client";

type AuthMode = "login" | "signup";

export function AuthForm({ initialMode }: { initialMode: AuthMode }) {
  const router = useRouter();
  const [mode, setMode] = useState<AuthMode>(initialMode);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    setError("");

    try {
      const supabase = createClient();
      if (mode === "login") {
        const { error: authError } = await supabase.auth.signInWithPassword({ email, password });
        if (authError) throw authError;
        router.push("/account");
        router.refresh();
        return;
      }

      const confirmationUrl = getAuthConfirmationUrl(
        window.location.origin,
        process.env.NEXT_PUBLIC_SITE_URL,
      );
      const { data, error: authError } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: { full_name: fullName.trim() || undefined },
          emailRedirectTo: confirmationUrl,
        },
      });
      if (authError) throw authError;

      if (data.session) {
        router.push("/onboarding");
        router.refresh();
        return;
      }

      setMessage("Account created. Check your email to confirm the account. The confirmation link will return you to CourtFlux and continue onboarding.");
      setMode("login");
    } catch {
      setError(
        mode === "login"
          ? "Unable to sign in. Check your email and password and try again."
          : "Unable to create an account with these details. If you already have an account, sign in instead."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 560, margin: "0 auto", paddingTop: 72 }}>
      <section className="panel" aria-labelledby="auth-title">
        <div style={{ marginBottom: 24 }}>
          <div className="muted" style={{ letterSpacing: "0.12em", textTransform: "uppercase" }}>CourtFlux</div>
          <h1 id="auth-title" style={{ marginBottom: 8 }}>{mode === "login" ? "Sign in" : "Create account"}</h1>
          <p className="muted">Basketball Intelligence from Video</p>
        </div>

        <form onSubmit={submit} aria-busy={busy}>
          <div className="form-grid">
            {mode === "signup" ? (
              <div className="field full">
                <label htmlFor="full-name">Full name</label>
                <input id="full-name" value={fullName} onChange={(event) => setFullName(event.target.value)} autoComplete="name" />
              </div>
            ) : null}
            <div className="field full">
              <label htmlFor="email">Email</label>
              <input id="email" type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required />
            </div>
            <div className="field full">
              <label htmlFor="password">Password</label>
              <input id="password" type="password" minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} required />
            </div>
          </div>

          {mode === "signup" ? (
            <p className="muted" style={{ marginTop: 14, fontSize: 13, lineHeight: 1.55 }}>
              By creating an account, you agree to the <Link href="/terms">Terms</Link> and acknowledge the <Link href="/privacy">Privacy Policy</Link>. CourtFlux will ask you to set up your role and first team before your first analysis.
            </p>
          ) : null}

          <div className="actions" style={{ marginTop: 20 }}>
            <button className="primary" type="submit" disabled={busy}>
              {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
            <button
              type="button"
              onClick={() => {
                setMode(mode === "login" ? "signup" : "login");
                setError("");
                setMessage("");
              }}
            >
              {mode === "login" ? "Create an account" : "I already have an account"}
            </button>
          </div>
        </form>

        <div aria-live="polite" aria-atomic="true">
          {error ? <div className="error-box" role="alert" style={{ marginTop: 18 }}>{error}</div> : null}
          {message ? <div className="muted" role="status" style={{ marginTop: 18 }}>{message}</div> : null}
        </div>

        <div style={{ marginTop: 22 }}>
          <Link href="/" className="muted">← Back to CourtFlux</Link>
        </div>
      </section>
    </main>
  );
}
