from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from supabase import create_client
except Exception:  # pragma: no cover - dependency is optional until worker install
    create_client = None


class SupabaseVideoStore:
    """Resolve private `supabase://bucket/path` video sources to a worker-local cache."""

    def __init__(self, output_dir: str = "data/uploads") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not self.url or not self.key or create_client is None:
            raise RuntimeError(
                "Supabase video download requires SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) "
                "and SUPABASE_SERVICE_ROLE_KEY on the worker"
            )
        self.client = create_client(self.url, self.key)

    @staticmethod
    def parse_source(source: str) -> tuple[str, str]:
        prefix = "supabase://"
        if not source.startswith(prefix):
            raise ValueError("Not a Supabase storage source")
        remainder = source[len(prefix):]
        bucket, separator, object_path = remainder.partition("/")
        if not separator or not bucket or not object_path:
            raise ValueError("Supabase source must be supabase://<bucket>/<object-path>")
        return bucket, object_path

    def get_local_video(self, source: str, expected_workspace_id: str | None = None) -> str:
        bucket, object_path = self.parse_source(source)
        if bucket != "analysis-videos":
            raise RuntimeError(f"Unsupported CourtFlux video bucket: {bucket}")

        if expected_workspace_id:
            expected_prefix = f"{expected_workspace_id}/"
            if not object_path.startswith(expected_prefix):
                raise RuntimeError("Uploaded video does not belong to the requested workspace")

        suffix = Path(object_path).suffix.lower()
        if suffix not in {".mp4", ".mov", ".webm", ".m4v"}:
            suffix = ".mp4"
        cache_name = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24] + suffix
        local_path = self.output_dir / cache_name
        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"⚡ Found cached uploaded video '{local_path}'.", flush=True)
            return str(local_path)

        signed = self.client.storage.from_(bucket).create_signed_url(object_path, 3600)
        signed_url = None
        if isinstance(signed, dict):
            signed_url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url")
        if not signed_url:
            raise RuntimeError("Could not create a signed URL for the uploaded CourtFlux video")

        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        print(f"📥 Downloading uploaded CourtFlux video to '{local_path}'.", flush=True)
        request = Request(str(signed_url), headers={"User-Agent": "CourtFlux-Worker/0.4"})
        try:
            with urlopen(request, timeout=120) as response, tmp_path.open("wb") as destination:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError("Uploaded video download completed without any bytes")
            tmp_path.replace(local_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return str(local_path)
