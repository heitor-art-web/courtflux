from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from core.p3_clip import P3ClipRequest


CLIP_GENERATOR_VERSION = "p3.clip_generation.v1"


class ClipGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class PhysicalClipConfig:
    version: str = CLIP_GENERATOR_VERSION
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    preset: str = "veryfast"
    crf: int = 23
    audio_bitrate: str = "128k"
    pixel_format: str = "yuv420p"

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "preset": self.preset,
            "crf": int(self.crf),
            "audio_bitrate": self.audio_bitrate,
            "pixel_format": self.pixel_format,
        }


@dataclass(slots=True)
class PhysicalClipArtifact:
    path: str
    size_bytes: int
    checksum_sha256: str
    content_type: str
    game_id: str
    analysis_run_id: str
    source_video_ref: str
    review_type: str
    source_start_ms: int
    source_end_ms: int
    clip_start_ms: int
    clip_end_ms: int
    source_event_ids: tuple[str, ...]
    source_possession_id: str | None
    game_player_ids: tuple[str, ...]
    context_policy_version: str
    request_contract_version: str
    request_fingerprint: str
    generation_version: str

    def metadata(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "analysis_run_id": self.analysis_run_id,
            "source_video_ref": self.source_video_ref,
            "review_type": self.review_type,
            "source_start_ms": self.source_start_ms,
            "source_end_ms": self.source_end_ms,
            "clip_start_ms": self.clip_start_ms,
            "clip_end_ms": self.clip_end_ms,
            "source_event_ids": list(self.source_event_ids),
            "source_possession_id": self.source_possession_id,
            "game_player_ids": list(self.game_player_ids),
            "context_policy_version": self.context_policy_version,
            "request_contract_version": self.request_contract_version,
            "request_fingerprint": self.request_fingerprint,
            "generation_version": self.generation_version,
        }


def resolve_ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # pragma: no cover - environment-specific dependency path
        raise ClipGenerationError("ffmpeg is unavailable") from exc


def probe_video_duration_ms(
    source_path: str | Path,
    *,
    ffprobe_path: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> int | None:
    """Best-effort source duration probe used to clamp planned review clips."""

    source = Path(source_path)
    if not source.exists() or not source.is_file() or source.stat().st_size <= 0:
        raise ClipGenerationError(f"source video is missing or empty: {source}")

    executable = ffprobe_path or shutil.which("ffprobe")
    if not executable:
        return None

    try:
        result = runner(
            [
                str(executable),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if int(getattr(result, "returncode", 1)) != 0:
        return None
    raw = str(getattr(result, "stdout", "") or "").strip()
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if seconds <= 0 or seconds != seconds or seconds == float("inf"):
        return None
    return max(1, int(round(seconds * 1000.0)))


def build_ffmpeg_command(
    request: P3ClipRequest,
    *,
    source_path: str | Path,
    output_path: str | Path,
    ffmpeg_path: str,
    config: PhysicalClipConfig | None = None,
) -> list[str]:
    config = config or PhysicalClipConfig()
    if request.generation_version != config.version:
        raise ValueError(
            "clip request generation_version does not match physical generator config"
        )

    duration_ms = int(request.clip_end_ms) - int(request.clip_start_ms)
    if duration_ms <= 0:
        raise ValueError("physical clip requires a positive duration")

    start_seconds = request.clip_start_ms / 1000.0
    duration_seconds = duration_ms / 1000.0
    return [
        str(ffmpeg_path),
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-sn",
        "-c:v",
        config.video_codec,
        "-preset",
        config.preset,
        "-crf",
        str(int(config.crf)),
        "-pix_fmt",
        config.pixel_format,
        "-c:a",
        config.audio_codec,
        "-b:a",
        config.audio_bitrate,
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_physical_clip(
    request: P3ClipRequest,
    *,
    source_path: str | Path,
    output_path: str | Path,
    config: PhysicalClipConfig | None = None,
    ffmpeg_path: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> PhysicalClipArtifact:
    """Generate one H.264/AAC MP4 evidence clip and atomically publish the file."""

    config = config or PhysicalClipConfig()
    source = Path(source_path)
    output = Path(output_path)
    if not source.exists() or not source.is_file() or source.stat().st_size <= 0:
        raise ClipGenerationError(f"source video is missing or empty: {source}")
    if source.resolve() == output.resolve():
        raise ClipGenerationError("source and output clip paths must be different")
    if output.suffix.lower() != ".mp4":
        raise ClipGenerationError("physical evidence clips must use an .mp4 output path")

    output.parent.mkdir(parents=True, exist_ok=True)
    working = output.with_name(f"{output.stem}.working{output.suffix}")
    working.unlink(missing_ok=True)

    executable = ffmpeg_path or resolve_ffmpeg_executable()
    command = build_ffmpeg_command(
        request,
        source_path=source,
        output_path=working,
        ffmpeg_path=executable,
        config=config,
    )

    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        working.unlink(missing_ok=True)
        raise ClipGenerationError(f"ffmpeg execution failed: {exc}") from exc

    if int(getattr(result, "returncode", 1)) != 0:
        working.unlink(missing_ok=True)
        detail = str(
            getattr(result, "stderr", "") or "ffmpeg returned no error output"
        ).strip()
        raise ClipGenerationError(f"ffmpeg clip generation failed: {detail}")
    if not working.exists() or working.stat().st_size <= 0:
        working.unlink(missing_ok=True)
        raise ClipGenerationError("ffmpeg completed without producing clip bytes")

    working.replace(output)
    return PhysicalClipArtifact(
        path=str(output),
        size_bytes=output.stat().st_size,
        checksum_sha256=_checksum(output),
        content_type="video/mp4",
        game_id=request.game_id,
        analysis_run_id=request.analysis_run_id,
        source_video_ref=request.source_video_ref,
        review_type=request.review_type,
        source_start_ms=request.source_start_ms,
        source_end_ms=request.source_end_ms,
        clip_start_ms=request.clip_start_ms,
        clip_end_ms=request.clip_end_ms,
        source_event_ids=tuple(request.source_event_ids),
        source_possession_id=request.source_possession_id,
        game_player_ids=tuple(request.game_player_ids),
        context_policy_version=request.context_policy_version,
        request_contract_version=request.contract_version,
        request_fingerprint=request.fingerprint(),
        generation_version=config.version,
    )
