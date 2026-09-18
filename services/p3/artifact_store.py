from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from services.p3.clip_generator import PhysicalClipArtifact


P3_ARTIFACT_BUCKET = "analysis-artifacts"
P3_CLIP_ARTIFACT_KIND = "p3_evidence_clip"
P3_CLIP_STORAGE_CONTRACT_VERSION = "p3.clip_storage.v1"


@dataclass(slots=True)
class StoredClipArtifact:
    artifact_id: str
    bucket: str
    object_path: str
    content_type: str
    size_bytes: int
    checksum_sha256: str


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class P3ArtifactStore:
    """Durably persist physical P3 clips in private Supabase Storage.

    Object paths are content-addressed by checksum. Repeating the same upload is therefore
    safe: an upsert may replace bytes only with bytes whose local checksum is identical to
    the path identity. Browser clients never receive direct write access to this bucket.
    """

    def __init__(
        self,
        client: Any,
        *,
        bucket: str = P3_ARTIFACT_BUCKET,
        kind: str = P3_CLIP_ARTIFACT_KIND,
    ) -> None:
        self.client = client
        self.bucket = str(bucket)
        self.kind = str(kind)

    @staticmethod
    def _path_component(value: str, *, name: str) -> str:
        component = str(value).strip()
        if not component or "/" in component or "\\" in component or component in {".", ".."}:
            raise ValueError(f"{name} is not a safe Storage path component")
        return component

    @classmethod
    def object_path(
        cls,
        *,
        workspace_id: str,
        analysis_run_id: str,
        clip_id: str,
        checksum_sha256: str,
    ) -> str:
        workspace = cls._path_component(workspace_id, name="workspace_id")
        run = cls._path_component(analysis_run_id, name="analysis_run_id")
        clip = cls._path_component(clip_id, name="clip_id")
        checksum = str(checksum_sha256).strip().lower()
        if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
            raise ValueError("clip checksum must be a lowercase/hex SHA-256 digest")
        return f"{workspace}/{run}/p3/clips/{clip}/{checksum}.mp4"

    def persist_clip(
        self,
        *,
        workspace_id: str,
        analysis_run_id: str,
        clip_id: str,
        artifact: PhysicalClipArtifact,
    ) -> StoredClipArtifact:
        if artifact.analysis_run_id != str(analysis_run_id):
            raise RuntimeError("physical clip analysis_run_id does not match artifact destination")
        if artifact.content_type != "video/mp4":
            raise RuntimeError("P3 evidence clips must be stored as video/mp4")

        path = Path(artifact.path)
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"physical clip is missing or empty: {path}")

        actual_size = path.stat().st_size
        actual_checksum = _file_sha256(path)
        if actual_size != int(artifact.size_bytes):
            raise RuntimeError(
                "physical clip size changed after generation "
                f"({artifact.size_bytes} != {actual_size})"
            )
        if actual_checksum != str(artifact.checksum_sha256).lower():
            raise RuntimeError("physical clip checksum changed after generation")

        object_path = self.object_path(
            workspace_id=workspace_id,
            analysis_run_id=analysis_run_id,
            clip_id=clip_id,
            checksum_sha256=actual_checksum,
        )

        with path.open("rb") as handle:
            self.client.storage.from_(self.bucket).upload(
                path=object_path,
                file=handle,
                file_options={
                    "cache-control": "31536000",
                    "content-type": artifact.content_type,
                    # Content-addressed paths make retries safe and idempotent.
                    "upsert": "true",
                },
            )

        metadata = {
            **artifact.metadata(),
            "clip_id": str(clip_id),
            "workspace_id": str(workspace_id),
            "checksum_algorithm": "sha256",
            "storage_contract_version": P3_CLIP_STORAGE_CONTRACT_VERSION,
        }
        row = {
            "run_id": str(analysis_run_id),
            "kind": self.kind,
            "bucket": self.bucket,
            "object_path": object_path,
            "content_type": artifact.content_type,
            "size_bytes": actual_size,
            "checksum": actual_checksum,
            "metadata": metadata,
        }
        result = self.client.table("analysis_artifacts").upsert(
            row,
            on_conflict="run_id,kind,object_path",
        ).execute()

        artifact_id: str | None = None
        data = getattr(result, "data", None)
        if isinstance(data, list) and data:
            candidate = data[0]
            if isinstance(candidate, dict) and candidate.get("id"):
                artifact_id = str(candidate["id"])
        elif isinstance(data, dict) and data.get("id"):
            artifact_id = str(data["id"])

        if artifact_id is None:
            lookup = (
                self.client.table("analysis_artifacts")
                .select("id")
                .eq("run_id", str(analysis_run_id))
                .eq("kind", self.kind)
                .eq("object_path", object_path)
                .limit(1)
                .execute()
            )
            rows = getattr(lookup, "data", None) or []
            if rows and isinstance(rows[0], dict) and rows[0].get("id"):
                artifact_id = str(rows[0]["id"])

        if not artifact_id:
            raise RuntimeError("analysis_artifacts upsert returned no durable artifact id")

        return StoredClipArtifact(
            artifact_id=artifact_id,
            bucket=self.bucket,
            object_path=object_path,
            content_type=artifact.content_type,
            size_bytes=actual_size,
            checksum_sha256=actual_checksum,
        )
