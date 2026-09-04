import { createAdminClient } from "@/lib/supabase/admin";
import { createClient } from "@/lib/supabase/server";

export const dynamic = "force-dynamic";

const VIDEO_BUCKET = "analysis-videos";
const MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set(["mp4", "mov", "webm", "m4v"]);
const ALLOWED_CONTENT_TYPES = new Set([
  "video/mp4",
  "video/quicktime",
  "video/webm",
  "video/x-m4v",
  "application/octet-stream",
  "",
]);

function extensionFromName(name: string) {
  const clean = name.trim().toLowerCase();
  const extension = clean.includes(".") ? clean.split(".").pop() ?? "" : "";
  return ALLOWED_EXTENSIONS.has(extension) ? extension : null;
}

export async function POST(request: Request) {
  try {
    const supabase = await createClient();
    const {
      data: { user },
    } = await supabase.auth.getUser();

    if (!user) {
      return Response.json({ error: "Authentication required" }, { status: 401 });
    }

    const { data: membership, error: membershipError } = await supabase
      .from("workspace_members")
      .select("workspace_id, role")
      .eq("user_id", user.id)
      .eq("status", "active")
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();

    if (membershipError) throw membershipError;
    if (!membership?.workspace_id) {
      return Response.json({ error: "No active CourtFlux workspace" }, { status: 403 });
    }
    if (!new Set(["owner", "admin", "analyst"]).has(membership.role)) {
      return Response.json({ error: "Your workspace role cannot start analyses" }, { status: 403 });
    }

    const body = (await request.json()) as {
      file_name?: unknown;
      file_size?: unknown;
      content_type?: unknown;
    };
    const fileName = typeof body.file_name === "string" ? body.file_name.trim() : "";
    const fileSize = Number(body.file_size);
    const contentType = typeof body.content_type === "string" ? body.content_type.trim().toLowerCase() : "";
    const extension = extensionFromName(fileName);

    if (!fileName || !extension) {
      return Response.json({ error: "Upload an MP4, MOV, WEBM or M4V basketball video" }, { status: 400 });
    }
    if (!Number.isFinite(fileSize) || fileSize <= 0 || fileSize > MAX_FILE_BYTES) {
      return Response.json({ error: "Video must be larger than 0 bytes and no more than 2 GB" }, { status: 400 });
    }
    if (!ALLOWED_CONTENT_TYPES.has(contentType)) {
      return Response.json({ error: "Unsupported video content type" }, { status: 400 });
    }

    const workspaceId = membership.workspace_id as string;
    const objectPath = `${workspaceId}/${user.id}/${crypto.randomUUID()}.${extension}`;
    const admin = createAdminClient();
    const { data, error } = await admin.storage.from(VIDEO_BUCKET).createSignedUploadUrl(objectPath);
    if (error || !data?.token) {
      throw error ?? new Error("Could not create video upload token");
    }

    return Response.json({
      bucket: VIDEO_BUCKET,
      path: objectPath,
      token: data.token,
      source: `supabase://${VIDEO_BUCKET}/${objectPath}`,
      max_file_bytes: MAX_FILE_BYTES,
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
