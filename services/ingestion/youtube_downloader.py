import os
from pathlib import Path
import yt_dlp


class YouTubeDownloader:

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_local_video(self, url: str, force: bool = False) -> str:
        """
        Resolve a YouTube URL to a cached local MP4.

        The analytics pipeline does not need audio, so the downloader prefers an MP4
        video stream. Caching by video ID makes every beta run use the exact same local
        input after the first successful download.
        """
        print("🔍 Checking YouTube video metadata...", flush=True)

        metadata_opts = {
            "quiet": True,
            "no_warnings": False,
            "noplaylist": True,
            "retries": 3,
        }

        try:
            with yt_dlp.YoutubeDL(metadata_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise RuntimeError(
                "YouTube metadata extraction failed. Update yt-dlp and check its console output "
                f"for the exact YouTube extractor error. Original error: {exc}"
            ) from exc

        video_id = info.get("id", "video")
        title = info.get("title", "Unknown title")
        duration = info.get("duration")
        print(
            f"🎞️ YouTube: {title} | id={video_id} | duration={duration or '?'}s",
            flush=True,
        )

        local_path = Path(self.output_dir) / f"{video_id}.mp4"

        if local_path.exists() and local_path.stat().st_size > 0 and not force:
            print(f"⚡ Found cached video '{local_path}'. Skipping download.", flush=True)
            return str(local_path)

        print(f"📥 Downloading video '{video_id}' from YouTube...", flush=True)
        ydl_opts = {
            "format": "bestvideo[ext=mp4]/best[ext=mp4]/bestvideo/best",
            "outtmpl": str(local_path),
            "noplaylist": True,
            "retries": 3,
            "fragment_retries": 3,
            "quiet": False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as exc:
            raise RuntimeError(f"YouTube download failed: {exc}") from exc

        if not local_path.exists() or local_path.stat().st_size == 0:
            raise RuntimeError(f"yt-dlp completed but the expected file was not created: {local_path}")

        print(f"✅ Download complete: {local_path}", flush=True)
        return str(local_path)
