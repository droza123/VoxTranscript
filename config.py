import os
from utils import resource_path

# Base directory for models
MODELS_DIR = resource_path("models")

# Model paths
FASTER_WHISPER_PATH = resource_path(os.path.join("models", "faster_whisper"))
ALIGN_MODEL_DIR = resource_path(os.path.join("models", "wav2vec2_base"))
VAD_MODEL_FP = resource_path(os.path.join("models", "voice_activity_detection", "whisperx_vad_segmentation.bin"))
PYANNOTE_CONFIG_PATH = resource_path("pyannote_config.yaml")

# Language mapping remains unchanged
LANGUAGE_MAP = {
    "Automatic": None,
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", 
    "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Russian": "ru", 
    "Chinese": "zh", "Japanese": "ja"
}