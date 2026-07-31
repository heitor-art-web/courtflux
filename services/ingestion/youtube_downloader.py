import os
import yt_dlp


class YouTubeDownloader:

  def __init__(self, output_dir: str = "data"):
    self.output_dir = output_dir
    os.makedirs(self.output_dir, exist_ok=True)

  def get_local_video(self, url: str, force: bool = False) -> str:
    """Extracts YouTube ID to cache videos individually by ID."""
    print("🔍 Checking video metadata...")
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
      info = ydl.extract_info(url, download=False)
      video_id = info.get("id", "video")

    local_path = os.path.join(self.output_dir, f"{video_id}.mp4")

    # If file exists and force isn't set, use cached file
    if os.path.exists(local_path) and not force:
      print(f"⚡ Found cached video '{local_path}'. Skipping download.")
      return local_path

    print(f"📥 Downloading video '{video_id}' from YouTube...")
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": local_path,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      ydl.download([url])

    print("✅ Download complete.")
    return local_path