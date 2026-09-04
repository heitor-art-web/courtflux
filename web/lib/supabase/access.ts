import { createClient } from "@/lib/supabase/server";

export async function canAccessJob(jobId: string) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();

  if (!user) return { allowed: false as const, status: 401, error: "Authentication required" };

  const { data, error } = await supabase
    .from("analysis_jobs")
    .select("id")
    .eq("id", jobId)
    .maybeSingle();

  if (error) throw error;
  if (!data) return { allowed: false as const, status: 404, error: "Job not found" };
  return { allowed: true as const, user };
}
