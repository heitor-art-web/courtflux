"use client";

import { FormEvent, useState } from "react";
import { createClient } from "@/lib/supabase/client";

export default function LoginPage() {
  const [mode, setMode] = useState<"login" | "signup">("login");
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
        window.location.assign("/");
        return;
      }

      const { data, error: authError } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: { full_name: fullName.trim() || undefined },
        },
      });
      if (authError) throw authError;

      if (data.session) {
        window.location.assign("/");
        return;
      }
      setMessage("Account created. Check your email to confirm the account, then sign in.");
      setMode("login");
    } catch (authError) {
      setError(authError instanceof Error ? authError.message : String(authError));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 560, margin: "0 auto", paddingTop: 72 }}>
      <section className="panel">
        <div style={{ marginBottom: 24 }}>
          <div className="muted" style={{ letterSpacing: "0.12em", textTransform: "uppercase" }}>CourtFlux</div>
          <h1 style={{ marginBottom: 8 }}>{mode === "login" ? "Sign in" : "Create account"}</h1>
          <p className="muted">Basketball Intelligence from Video</p>
        </div>

        <form onSubmit={submit}>
          <div className="form-grid">
            {mode === "signup" ? (
              <div className="field full">
                <label>Full name</label>
                <input value={fullName} onChange={(event) => setFullName(event.target.value)} autoComplete="name" />
              </div>
            ) : null}
            <div className="field full">
              <label>Email</label>
              <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required />
            </div>
            <div className="field full">
              <label>Password</label>
              <input type="password" minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} required />
            </div>
          </div>

          <div className="actions" style={{ marginTop: 20 }}>
            <button className="primary" type="submit" disabled={busy}>
              {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
            <button type="button" onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(""); setMessage(""); }}>
              {mode === "login" ? "Create an account" : "I already have an account"}
            </button>
          </div>
        </form>

        {error ? <div className="error-box" style={{ marginTop: 18 }}>{error}</div> : null}
        {message ? <div className="muted" style={{ marginTop: 18 }}>{message}</div> : null}
      </section>
    </main>
  );
}
