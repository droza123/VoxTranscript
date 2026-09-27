# VoxTranscript

A Windows desktop application for offline audio transcription with speaker diarization, built on top of WhisperX, faster-whisper, and pyannote.audio.

## Installation

Download the latest installer from the [Releases page](https://github.com/droza123/VoxTranscript/releases) and run `VoxTranscript_Setup.exe`. The installer requires no administrator privileges and installs to `%LOCALAPPDATA%\VoxTranscript\`.

The installer is split into multiple `.bin` slices to fit GitHub's per-file size limit. Download all files (`VoxTranscript_Setup.exe` and every `VoxTranscript_Setup-*.bin`) into the same folder before running the `.exe`.

## Development setup

Requirements: Python 3.10, [Inno Setup 7](https://jrsoftware.org/isdl.php) (for building the installer).

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

### Models

The bundled installer ships with the models below pre-installed. For a development checkout, download each into its corresponding subfolder under `models/`:

- **faster-whisper** → `models/faster_whisper/` — download from [Systran on HuggingFace](https://huggingface.co/Systran)
- **pyannote segmentation 3.0** → `models/pyannote/segmentation-3.0/` — download from [HuggingFace](https://huggingface.co/pyannote/segmentation-3.0) (gated, requires accepting terms)
- **pyannote wespeaker-voxceleb-resnet34-LM** → `models/pyannote/wespeaker-voxceleb-resnet34-LM/` — download from [HuggingFace](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) (gated, requires accepting terms)
- **wav2vec2** → `models/wav2vec2_base/` — download from [download.pytorch.org/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth](https://download.pytorch.org/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth)
- **SpeechBrain ECAPA-TDNN** → `models/speechbrain_models/` — downloaded automatically on first run

### Building the installer

The summary prompts are built in per organisation (see `prompt_profiles.py`). Pick one with `VOXTRANSCRIPT_PROFILE`; it defaults to `legion`:

```powershell
# Legionaries of Christ (Spanish prompts)
$env:VOXTRANSCRIPT_PROFILE="legion"
pyinstaller app.spec --clean
& "C:\Program Files\Inno Setup 7\ISCC.exe" VoxTranscriptSetup.iss

# Oblates (Italian prompts, Madre Maria Elisabetta Patrizi)
$env:VOXTRANSCRIPT_PROFILE="oblates"
pyinstaller app.spec --clean
& "C:\Program Files\Inno Setup 7\ISCC.exe" VoxTranscriptSetup.iss
```

PyInstaller prints `*** Building VoxTranscript with the '<profile>' summary prompts ***` near the start. Output lands in `Output\` as `VoxTranscript_Setup_<profile>.exe` plus one or more `.bin` slice files; the name comes from the profile the app in `dist\` was built with. When running from source, `$env:VOXTRANSCRIPT_PROFILE` selects the prompts too.

## Acknowledgements

VoxTranscript bundles and uses the following third-party models and software. Their LICENSE and README files are preserved inside the installed `models/` directory.

| Component | Authors | License | Source |
|---|---|---|---|
| **WhisperX** (transcription pipeline) | Max Bain et al. | BSD-4-Clause | [m-bain/whisperX](https://github.com/m-bain/whisperX) |
| **faster-whisper** (CTranslate2 Whisper inference) | Systran | MIT | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| **OpenAI Whisper** (underlying speech recognition model) | OpenAI | MIT | [openai/whisper](https://github.com/openai/whisper) |
| **pyannote.audio** (speaker diarization toolkit) | Hervé Bredin / CNRS | MIT | [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) |
| **pyannote/segmentation-3.0** (speaker segmentation model) | Alexis Plaquet, Hervé Bredin | MIT | [HF: pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) |
| **pyannote/wespeaker-voxceleb-resnet34-LM** (speaker embedding model) | Wenet/WeSpeaker team; pyannote.audio wrapper by H. Bredin | CC-BY 4.0 (inherited from [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)) | [HF: pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) |
| **wav2vec2** (forced-alignment model) | Meta AI / Facebook AI Research | BSD-style (via torchaudio) | [pytorch/audio](https://github.com/pytorch/audio) |
| **SpeechBrain `spkrec-ecapa-voxceleb`** (speaker embedding model) | SpeechBrain team | Apache 2.0 | [HF: speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| **FFmpeg** (audio decoding) | FFmpeg contributors | LGPL/GPL | [ffmpeg.org](https://ffmpeg.org/) |

### Citations

If you use VoxTranscript in academic work, please also cite the underlying research:

```bibtex
@inproceedings{bain2023whisperx,
  title={WhisperX: Time-Accurate Speech Transcription of Long-Form Audio},
  author={Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
  booktitle={INTERSPEECH 2023},
  year={2023}
}

@inproceedings{Plaquet23,
  title={Powerset multi-class cross entropy loss for neural speaker diarization},
  author={Plaquet, Alexis and Bredin, Hervé},
  booktitle={Proc. INTERSPEECH 2023},
  year={2023}
}

@inproceedings{Bredin23,
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  author={Bredin, Hervé},
  booktitle={Proc. INTERSPEECH 2023},
  year={2023}
}

@inproceedings{Wang2023,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={ICASSP 2023},
  year={2023}
}
```

Each bundled model also retains its own LICENSE and README inside the install's `models/` subdirectory.
