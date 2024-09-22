# gui/settings.py

import json
import os
import logging
import torch
from utils import get_app_file_path

SETTINGS_FILE = 'whisper_gui_settings.json'

class SettingsManager:
    def __init__(self):
        self.settings_path = get_app_file_path(SETTINGS_FILE, 'config')
        self.settings = self.load_settings()
        self.changed = False

    def load_settings(self):
        logging.info(f"Loading settings from file: {self.settings_path}")
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
                    logging.info(f"Settings loaded successfully from {self.settings_path}")
                    return settings
            except json.JSONDecodeError:
                logging.error(f"Failed to parse {self.settings_path}. Using default settings.")
        else:
            logging.info(f"Settings file not found at {self.settings_path}. Using default settings.")
        return self.get_default_settings()

    def set_defaults(self, settings):
        defaults = self.get_default_settings()
        for key, value in defaults.items():
            settings.setdefault(key, value)

    def get_default_settings(self):
        return {
            'save_to_media_folder': True,
            'output_folder': '',
            'output_formats': ['txt', 'srt', 'jsonl'],
            'model': 'large-v2',
            'language': 'Automatic',
            'diarize': True,
            'speaker_count': 'Automatic',
            'use_custom_filenames': False,
            'filename_prefix': '',
            'filename_suffix': '',
            'use_gpu': torch.cuda.is_available(),
            'last_file_directory': os.path.expanduser('~'),
            'last_folder_directory': os.path.expanduser('~'),
            'auto_summarize': False
        }

    def save_settings(self):
        if self.changed:
            logging.info(f"Saving settings to file: {self.settings_path}")
            with open(self.settings_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            logging.info(f"Settings saved successfully to {self.settings_path}")
            self.changed = False
        else:
            logging.info(f"No changes to save. Settings file remains unchanged: {self.settings_path}")

    def get(self, key, default=None):
        value = self.settings.get(key, default)
        if key == 'save_to_media_folder':
            value = bool(value)
        logging.debug(f"SettingsManager.get: {key} = {value}")
        return value

    def set(self, key, value):
        if self.settings.get(key) != value:
            self.settings[key] = value
            self.changed = True
            logging.info(f"Setting updated: {key} = {value}")

    def get_last_directory(self, key):
        directory = self.get(key)
        logging.debug(f"Retrieved directory for {key}: {directory}")
        if directory and os.path.exists(directory):
            logging.debug(f"Directory exists, returning: {directory}")
            return directory
        logging.debug(f"Directory not found or doesn't exist, returning home directory")
        return os.path.expanduser('~')
    
    def generate_output_filename(self, audio_file):
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        use_custom_filenames = self.get('use_custom_filenames', False)
        
        if use_custom_filenames:
            filename_prefix = self.get('filename_prefix', '')
            filename_suffix = self.get('filename_suffix', '')
            return f"{filename_prefix}{base_name}{filename_suffix}"
        else:
            return f"{base_name}_transcription"

    def get_output_folder(self, audio_file):
        save_to_media_folder = self.get('save_to_media_folder', True)
        return os.path.dirname(audio_file) if save_to_media_folder else self.get('output_folder', '')