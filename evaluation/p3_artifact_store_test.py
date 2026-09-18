from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from services.p3.artifact_store import P3ArtifactStore
from services.p3.clip_generator import PhysicalClipArtifact


class _Result:
    def __init__(self, data: Any = None) -> None:
        self.data = data


class _StorageBucket:
    def __init__(self, client: "_Client", bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    def upload(self, *, path: str, file: Any, file_options: dict[str, Any]) -> _Result:
        self.client.uploads.append(
            {
                "bucket": self.bucket,
                "path": path,
                "bytes": file.read(),
                "options": dict(file_options),
            }
        )
        return _Result({"path": path})


class _Storage:
    def __init__(self, client: "_Client") -> None:
        self.client = client

    def from_(self, bucket: str) -> _StorageBucket:
        return _StorageBucket(self.client, bucket)


class _Query:
    def __init__(self, client: "_Client", operation: str, payload: Any = None) -> None:
        self.client = client
        self.operation = operation
        self.payload = payload
        self.filters: list[tuple[str, str]] = []

    def eq(self, key: str, value: str) -> "_Query":
        self.filters.append((key, value))
        return self

    def limit(self, _value: int) -> "_Query":
        return self

    def execute(self) -> _Result:
        self.client.calls.append(
            {
                "operation": self.operation,
                "payload": self.payload,
                "filters": list(self.filters),
            }
        )
        if self.operation == "upsert:analysis_artifacts":
            return _Result([{"id": "artifact-1"}])
        return _Result([])


class _Table:
    def __init__(self, client: "_Client", name: str) -> None:
        self.client = client
        self.name = name

    def upsert(self, payload: Any, *, on_conflict: str) -> _Query:
        query = _Query(self.client, f"upsert:{self.name}", payload)
        self.client.conflicts.append(on_conflict)
        return query

    def select(self, _fields: str) -> _Query:
        return _Query(self.client, f"select:{self.name}")


class _Client:
    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.conflicts: list[str] = []
        self.storage = _Storage(self)

    def table(self, name: str) -> _Table:
        return _Table(self, name)


def _artifact(path: Path) -> PhysicalClipArtifact:
    checksum = sha256(path.read_bytes()).hexdigest()
    return PhysicalClipArtifact(
        path=str(path),
        size_bytes=path.stat().st_size,
        checksum_sha256=checksum,
        content_type="video/mp4",
        generation_version="p3.clip_generation.v1",
        clip_start_ms=1000,
        clip_end_ms=5000,
        source_event_ids=("shot-1",),
    )


def test_clip_upload_is_content_addressed_and_registered() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        path.write_bytes(b"courtflux-clip-bytes")
        client = _Client()
        store = P3ArtifactStore(client)

        result = store.persist_clip(
            workspace_id="workspace-1",
            analysis_run_id="run-1",
            clip_id="clip-1",
            artifact=_artifact(path),
        )

        assert result.artifact_id == "artifact-1"
        assert result.bucket == "analysis-artifacts"
        assert result.object_path.startswith("workspace-1/run-1/p3/clips/clip-1/")
        assert result.object_path.endswith(f"{result.checksum_sha256}.mp4")
        assert len(client.uploads) == 1
        assert client.uploads[0]["options"]["upsert"] == "true"
        assert client.uploads[0]["options"]["content-type"] == "video/mp4"
        assert client.conflicts == ["run_id,kind,object_path"]

        row = client.calls[0]["payload"]
        assert row["kind"] == "p3_evidence_clip"
        assert row["checksum"] == result.checksum_sha256
        assert row["metadata"]["clip_id"] == "clip-1"
        assert row["metadata"]["storage_contract_version"] == "p3.clip_storage.v1"


def test_tampered_clip_is_rejected_before_upload() -> None:
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        path.write_bytes(b"first")
        artifact = _artifact(path)
        path.write_bytes(b"tampered")
        client = _Client()
        store = P3ArtifactStore(client)

        try:
            store.persist_clip(
                workspace_id="workspace-1",
                analysis_run_id="run-1",
                clip_id="clip-1",
                artifact=artifact,
            )
        except RuntimeError as exc:
            assert "changed after generation" in str(exc)
        else:
            raise AssertionError("tampered physical clip must not be uploaded")

        assert client.uploads == []


def test_storage_path_components_are_not_user_paths() -> None:
    try:
        P3ArtifactStore.object_path(
            workspace_id="../workspace",
            analysis_run_id="run-1",
            clip_id="clip-1",
            checksum_sha256="a" * 64,
        )
    except ValueError as exc:
        assert "safe Storage path component" in str(exc)
    else:
        raise AssertionError("unsafe path components must be rejected")