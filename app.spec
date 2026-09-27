# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import site
import lightning_fabric
import whisperx
import pytorch_lightning
import speechbrain
import blobfile

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

block_cipher = None

# site-packages of the environment running PyInstaller. Don't hard-code a path here:
# a stale one makes PyInstaller bundle packages from some other (older) venv.
venv_site_packages = os.path.join(sys.prefix, 'Lib', 'site-packages')

# Get the path to the speechbrain package
speechbrain_path = os.path.dirname(speechbrain.__file__)

# Collect all submodules of pyannote
pyannote_hidden_imports = collect_submodules('pyannote')

# Collect all data files of pyannote
pyannote_datas = collect_data_files('pyannote')

# Collect all data files of whisperx
whisperx_datas = collect_data_files('whisperx')

# Collect data files for lightning_fabric and pytorch_lightning
lightning_fabric_datas = collect_data_files('lightning_fabric')
pytorch_lightning_datas = collect_data_files('pytorch_lightning')

# Collect all data files and submodules of speechbrain
speechbrain_datas = collect_data_files('speechbrain')
speechbrain_submodules = collect_submodules('speechbrain')

# Get the path to the blobfile package
blobfile_path = os.path.dirname(blobfile.__file__)

# Add ffmpeg to the binaries
ffmpeg_dir = 'ffmpeg'
ffmpeg_files = [(os.path.join(ffmpeg_dir, file), 'ffmpeg') for file in os.listdir(ffmpeg_dir) if file.endswith('.exe')]

a = Analysis(
    ['app.py'],
    pathex=[venv_site_packages],
    binaries=ffmpeg_files,  # Include ffmpeg binaries here
    datas=[
        ('gui', 'gui'),
        ('icons', 'icons'),
        ('pyannote_config.yaml', '.'),
        ('config.py', '.'),
        ('transcriber.py', '.'),
        ('pyannote_diarization.py', '.'),
        ('utils.py', '.'),
        ('models', 'models'),
        (blobfile_path, 'blobfile'),
        (speechbrain_path, 'speechbrain'),
    ] + whisperx_datas + pyannote_datas + lightning_fabric_datas + pytorch_lightning_datas + 
    speechbrain_datas + copy_metadata('speechbrain'),
    hiddenimports=[
        'lightning_fabric',
        'lightning',
        'pytorch_lightning',
        'whisperx',
        'speechbrain',
        'blobfile',
    ] + pyannote_hidden_imports + speechbrain_submodules,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['whisper_gui_settings.json'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # This ensures --onedir behavior
    name='VoxTranscript',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icons\\voxtranscript-icon.ico'],
)

# This COLLECT call is what creates the distribution directory
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VoxTranscript',
)