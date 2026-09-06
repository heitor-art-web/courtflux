"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";

export function SignOutButton() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  return (
    <div>
      <button
        type="button"
        disabled={busy}
        aria-busy={busy}
        onClick={async () => {
          setBusy(true);
          setErrorMessage(null);

          const supabase = createClient();
          const { error } = await supabase.auth.signOut({ scope: "global" });

          if (error) {
            setErrorMessage("Sign out failed. Please try again.");
            setBusy(false);
            return;
          }

          router.push("/login");
          router.refresh();
        }}
      >
        {busy ? "Signing out…" : "Sign out"}
      </button>
      {errorMessage ? <span role="alert">{errorMessage}</span> : null}
    </div>
  );
}
