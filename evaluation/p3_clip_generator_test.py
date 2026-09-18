from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from core.p3_clip import P3ClipRequest
from services.p3.clip_generator import (
    ClipGenerationError,
    PhysicalClipConfig,
    build_ffmpeg_command,
    generate_physical_clip,
)


def _request() -> P3ClipRequest:
    return P3ClipRequest(
        game_id="game-1",
        analysis_run_id="run-1",
        review_type="shot",
        source_start_ms=10_000,
        source_end_ms=10_500,
        clip_start_ms=4_000,
        clip_end_ms=14_500,
        source_event_ids=("shot-1",),
        game_player_ids=("game_player_03",),
    )


def test_ffmpeg_command_uses_versioned_transcode_profile_and_source_time() -> None:
    command = build_ffmpeg_command(
        _request(),
        source_path="game.mp4",
        output_path="clip.mp4",
        ffmpeg_path="ffmpeg",
    )
    assert command[:5] == ["ffmpeg", "-y", "-loglevel", "error", "-ss"]
    assert command[5] == "4.000"
    assert command[6:8] == ["-i", "game.mp4"]
    assert command[8:10] == ["-t", "10.500"]
    assert "libx264" in command
    assert "yuv420p" in command
    assert "+faststart" in command
    assert command[-1] == "clip.mp4"


def test_generator_atomically_materializes_clip_and_returns_checksum() -> None:
    class _Result:
        returncode = 0
        stderr = ""

    def fake_runner(command, **kwargs):
        del kwargs
        Path(command[-1]).write_bytes(b"courtflux-clip")
        return _Result()

    with TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "game.mp4"
        output = Path(temp_dir) / "clips" / "shot.mp4"
        source.write_bytes(b"source-video")

        artifact = generate_physical_clip(
            _request(),
            source_path=source,
            output_path=output,
            ffmpeg_path="ffmpeg-test",
            runner=fake_runner,
        )

        assert output.read_bytes() == b"courtflux-clip"
        assert not output.with_name("shot.working.mp4").exists()
        assert artifact.size_bytes == len(b"courtflux-clip")
        assert artifact.checksum_sha256 == sha256(b"courtflux-clip").hexdigest()
        assert artifact.content_type == "video/mp4"
        assert artifact.metadata()["source_event_ids"] == ["shot-1"]


def test_generator_failure_does_not_publish_partial_clip() -> None:
    class _Result:
        returncode = 1
        stderr = "encoder failed"

    def fake_runner(command, **kwargs):
        del kwargs
        Path(command[-1]).write_bytes(b"partial")
        return _Result()

    with TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / "game.mp4"
        output = Path(temp_dir) / "clip.mp4"
        source.write_bytes(b"source-video")

        try:
            generate_physical_clip(
                _request(),
                source_path=source,
                output_path=output,
                ffmpeg_path="ffmpeg-test",
                runner=fake_runner,
            )
        except ClipGenerationError as exc:
            assert "encoder failed" in str(exc)
        else:
            raise AssertionError("failed ffmpeg run must raise ClipGenerationError")

        assert not output.exists()
        assert not output.with_name("clip.working.mp4").exists()


def test_generator_refuses_version_mismatch() -> None:
    config = PhysicalClipConfig(version="p3.clip_generation.v2")
    try:
        build_ffmpeg_command(
            _request(),
            source_path="game.mp4",
            output_path="clip.mp4",
            ffmpeg_path="ffmpeg",
            config=config,
        )
    except ValueError as exc:
        assert "generation_version" in str(exc)
    else:
        raise AssertionError("version mismatch must not silently produce a clip")


def main() -> None:
    test_ffmpeg_command_uses_versioned_transcode_profile_and_source_time()
    test_generator_atomically_materializes_clip_and_returns_checksum()
    test_generator_failure_does_not_publish_partial_clip()
    test_generator_refuses_version_mismatch()
    print("p3_clip_generator_test: PASS")


if __name__ == "__main__":
    main()
