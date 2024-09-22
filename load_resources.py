import os
import platform
import shutil
import requests
from tqdm import tqdm

from utils import resource_path

def get_ffmpeg():
    system = platform.system()
    if system == "Windows":
        return get_ffmpeg_windows()
    else:
        return "ffmpeg"  # Assume ffmpeg is in PATH for non-Windows systems

def get_ffmpeg_windows():
    ffmpeg_dir = resource_path("ffmpeg")
    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        url = 'https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-essentials_build.zip'
        ffmpeg_zip = os.path.join(ffmpeg_dir, "ffmpeg.zip")
        ffmpeg_extract_dir = os.path.join(ffmpeg_dir, "ffmpeg_extract")
        ffmpeg_exe = os.path.join(ffmpeg_extract_dir, "ffmpeg-6.0-essentials_build", "bin", "ffmpeg.exe")
        
        os.makedirs(ffmpeg_dir, exist_ok=True)
        download_with_progress_bar(url, ffmpeg_zip)
        shutil.unpack_archive(ffmpeg_zip, ffmpeg_extract_dir, "zip")
        shutil.move(ffmpeg_exe, ffmpeg_path)
        shutil.rmtree(ffmpeg_extract_dir)
        os.remove(ffmpeg_zip)
    return ffmpeg_path

def download_with_progress_bar(url: str, filename: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)