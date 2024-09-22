from subprocess_context import silent_subprocess
from typing import Optional, Union
import os
import yaml
import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from utils import resource_path, get_app_file_path
from whisperx.audio import SAMPLE_RATE, load_audio
import gc

class DiarizationPipeline:
    def __init__(
        self,
        config_path=resource_path("pyannote_config.yaml"),
        model_path=None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        with silent_subprocess():
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            elif isinstance(device, str):
                device = torch.device(device)

            self.device = device
            self.model = None
            self.load_model(config_path, model_path)

    def load_model(self, config_path, model_path):
        # Load and modify configuration
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)

        if model_path:
            # Update paths in the configuration
            segmentation_model = os.path.join(model_path, "pyannote/segmentation-3.0/pytorch_model.bin").replace('\\', '/')
            embedding_model = os.path.join(model_path, "pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin").replace('\\', '/')
            config['pipeline']['params']['segmentation'] = segmentation_model
            config['pipeline']['params']['embedding'] = embedding_model

        # Create a temporary configuration file
        temp_config_path = get_app_file_path('temp_pyannote_config.yaml', 'temp')
        with open(temp_config_path, 'w') as temp_config_file:
            yaml.dump(config, temp_config_file)

        # Load model from the temporary config file
        self.model = Pipeline.from_pretrained(temp_config_path, use_auth_token=False).to(self.device)

        # Remove the temporary config file
        os.remove(temp_config_path)

    def unload_model(self):
        if self.model is not None:
            cpu_device = torch.device("cpu")
            self.model.to(cpu_device)
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        gc.collect()

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
        with silent_subprocess():
            if isinstance(audio, str):
                audio = load_audio(audio)
            audio_data = {
                "waveform": torch.from_numpy(audio[None, :]),
                "sample_rate": SAMPLE_RATE,  # 16KHz
            }
            segments = self.model(
                audio_data,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diarize_df = pd.DataFrame(
                segments.itertracks(yield_label=True),
                columns=["segment", "label", "speaker"],
            )
            diarize_df["start"] = diarize_df["segment"].apply(lambda x: x.start)
            diarize_df["end"] = diarize_df["segment"].apply(lambda x: x.end)
            return diarize_df

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()

# Context manager for easy use
def get_diarization_pipeline(config_path=resource_path("pyannote_config.yaml"), model_path=None, device=None):
    return DiarizationPipeline(config_path, model_path, device)