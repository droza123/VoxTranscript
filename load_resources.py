import os
import platform
import shutil
import requests
from tqdm import tqdm
import sys

from utils import resource_path

def get_ffmpeg():
    system = platform.system()
    if system == "Windows":
        ffmpeg_path = get_ffmpeg_windows()
        # Add FFMPEG directory to PATH
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ['PATH']
        return ffmpeg_path
    else:
        return "ffmpeg"  # Assume ffmpeg is in PATH for non-Windows systems

def get_ffmpeg_windows():
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
        ffmpeg_dir = os.path.join(base_path, "ffmpeg")
        ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    else:
        # Running in development mode
        ffmpeg_dir = resource_path("ffmpeg")
        ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    
    # ffprobe.exe must sit next to ffmpeg.exe: audio.prepare_audio() calls ffmpeg.probe(),
    # which runs "ffprobe" from PATH (get_ffmpeg_windows adds this folder to PATH).
    ffprobe_path = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe")
    if not os.path.exists(ffmpeg_path) or not os.path.exists(ffprobe_path):
        url = 'https://github.com/GyanD/codexffmpeg/releases/download/6.0/ffmpeg-6.0-essentials_build.zip'
        ffmpeg_zip = os.path.join(ffmpeg_dir, "ffmpeg.zip")
        ffmpeg_extract_dir = os.path.join(ffmpeg_dir, "ffmpeg_extract")
        extracted_bin = os.path.join(ffmpeg_extract_dir, "ffmpeg-6.0-essentials_build", "bin")
        
        os.makedirs(ffmpeg_dir, exist_ok=True)
        download_with_progress_bar(url, ffmpeg_zip)
        shutil.unpack_archive(ffmpeg_zip, ffmpeg_extract_dir, "zip")
        for exe_name, dest in (("ffmpeg.exe", ffmpeg_path), ("ffprobe.exe", ffprobe_path)):
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(os.path.join(extracted_bin, exe_name), dest)
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