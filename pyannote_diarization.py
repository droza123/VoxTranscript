from subprocess_context import silent_subprocess
from typing import Optional, Union
import os
import logging
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
        if model_path is None:
            model_path = resource_path("models")
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

        logging.info(f"Loading model with config_path: {config_path}, model_path: {model_path}")
        logging.info(f"Config contents: {config}")

        if model_path:
            # Update paths in the configuration
            segmentation_model = resource_path(os.path.join("models", "pyannote", "segmentation-3.0", "pytorch_model.bin"))
            embedding_model = resource_path(os.path.join("models", "pyannote", "wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin"))
            
            # Check if files exist
            if not os.path.exists(segmentation_model):
                raise FileNotFoundError(f"Segmentation model not found at {segmentation_model}")
            if not os.path.exists(embedding_model):
                raise FileNotFoundError(f"Embedding model not found at {embedding_model}")

            config['pipeline']['params']['segmentation'] = segmentation_model
            config['pipeline']['params']['embedding'] = embedding_model

        logging.info(f"Updated segmentation path: {segmentation_model}")
        logging.info(f"Updated embedding path: {embedding_model}")
        
        try:
            # Create a temporary configuration file
            temp_config_path = get_app_file_path('temp_pyannote_config.yaml', 'temp')
            try:
                with open(temp_config_path, 'w') as temp_config_file:
                    yaml.dump(config, temp_config_file)

                # Load model from the temporary config file
                self.model = Pipeline.from_pretrained(temp_config_path, use_auth_token=False).to(self.device)
                logging.info("Diarization model loaded successfully")
            finally:
                # Remove the temporary config file
                if os.path.exists(temp_config_path):
                    os.remove(temp_config_path)
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

        return self.model

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