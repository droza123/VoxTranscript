## Model Setup

This project requires several pre-trained models which are not included in the repository due to their size and licensing. Please follow these steps to set up the required models:

1. Create a `models` folder in the project root if it doesn't exist.
2. Download the following models and place them in the corresponding subfolders:

   - Faster Whisper models: Download from [HuggingFace](https://huggingface.co/guillaumekln) and place in `models/faster_whisper/`
   - Pyannote models: Download from [HuggingFace](https://huggingface.co/pyannote) and place in `models/pyannote/`
   - WAV2VEC2 model: Download from [PyTorch](https://download.pytorch.org/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth) and place in `models/wav2vec2_base/`
   - Voice Activity Detection model: Download from [WhisperX](https://github.com/m-bain/whisperX) and place in `models/voice_activity_detection/`
   - SpeechBrain models: These will be downloaded automatically on first run

Please ensure you comply with the licensing terms of each model.