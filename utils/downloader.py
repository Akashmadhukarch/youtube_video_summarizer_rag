import yt_dlp
import os


def download_audio(url: str, output_path: str = "audio"):
    os.makedirs(output_path, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}/%(id)s.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Get the actual file extension
        file_ext = info.get('ext', 'mp3')
        file_path = f"{output_path}/{info['id']}.{file_ext}"

    return file_path