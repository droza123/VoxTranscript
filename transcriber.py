"""
Wrapper built around the WhisperX library to transcribe audio files in OFFLINE mode.
The transcriber can be used to transcribe audio files, align the transcriptions, diarize the transcriptions.

pre-trained models required:
- faster-whisper
- pyannote diairzation 3.1
- wav2vec2-base: for alignment
    - Download link - https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth
- VAD(Voice Activity Detection) model
    
whisperx version: 3.1.3
"""

from subprocess_context import silent_subprocess
import gc
import os
import sys
import logging

import numpy as np
import torch
import gc
from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.diarize import assign_word_speakers
from whisperx.utils import get_writer
from utils import resource_path
from langdetect import detect, LangDetectException
from contextlib import contextmanager

from load_resources import get_ffmpeg
from audio import decode_audio
from pyannote_diarization import DiarizationPipeline


@contextmanager
def gpu_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logging.info("GPU memory managed and garbage collected")
        
class TranscriptionConfig:
    def __init__(
        self,
        whisper_model_name,
        whisper_download_root=None,
        device=None,
        device_index=0,
        batch_size=8,
        compute_type="float16",
        output_dir=".",
        output_format="all",
        verbose=True,
        task="transcribe",
        language=None,
        align_model=None,
        interpolate_method="nearest",
        no_align=False,
        return_char_alignments=False,
        vad_onset=0.500,
        vad_offset=0.363,
        chunk_size=30,
        diarize=False,
        min_speakers=None,
        max_speakers=None,
        save_voice_recognition=True,
        auto_summarize=False,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1.0,
        length_penalty=1.0,
        suppress_tokens="-1",
        suppress_numerals=False,
        initial_prompt=None,
        condition_on_previous_text=False,
        fp16=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        max_line_width=None,
        max_line_count=None,
        highlight_words=False,
        segment_resolution="sentence",
        threads=0,
        hf_token=None,
        print_progress=False,
        align_model_dir=None,
        vad_model_fp=None,
        pyannote_config_path=None,
        model_path=None,
    ):
        self.whisper_model_name = whisper_model_name
        self.whisper_download_root = whisper_download_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_index = device_index
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.output_dir = output_dir
        self.output_format = output_format
        self.verbose = verbose
        self.task = task
        self.language = language
        self.align_model = align_model
        self.interpolate_method = interpolate_method
        self.no_align = no_align
        self.return_char_alignments = return_char_alignments
        self.vad_onset = vad_onset
        self.vad_offset = vad_offset
        self.chunk_size = chunk_size
        self.diarize = diarize
        self.save_voice_recognition = save_voice_recognition
        self.auto_summarize = auto_summarize
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.temperature = temperature
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.suppress_tokens = suppress_tokens
        self.suppress_numerals = suppress_numerals
        self.initial_prompt = initial_prompt
        self.condition_on_previous_text = condition_on_previous_text
        self.fp16 = fp16
        self.temperature_increment_on_fallback = temperature_increment_on_fallback
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.max_line_width = max_line_width
        self.max_line_count = max_line_count
        self.highlight_words = highlight_words
        self.segment_resolution = segment_resolution
        self.threads = threads
        self.hf_token = hf_token
        self.print_progress = print_progress
        self.align_model_dir = align_model_dir
        self.vad_model_fp = vad_model_fp
        self.pyannote_config_path = pyannote_config_path
        self.model_path = model_path

class ProgressStream:
    def __init__(self, callback):
        self.callback = callback
        self.buffer = ""

    def write(self, text):
        if text.startswith('Progress:'):
            try:
                # Extract the progress value as a string, including the '%' symbol
                progress_text = text.split(':')[1].strip().rstrip('.')
                if self.callback:
                    try:
                        self.callback(progress_text)
                    except Exception as e:
                        logging.error(f"ProgressStream callback failed: {str(e)}")
                else:
                    logging.warning("ProgressStream callback is None")
            except Exception as e:
                logging.error(f"ProgressStream unexpected error: {str(e)}")
        
        # Accumulate text in the buffer
        self.buffer += text
        
        # If we have a complete line, log it and clear the buffer
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                logging.info(line)
            self.buffer = lines[-1]

    def flush(self):
        if self.buffer:
            logging.info(self.buffer)
            self.buffer = ""

class Transcriber:
    def __init__(self, config: TranscriptionConfig, settings_manager):
        self.config = config
        self.settings_manager = settings_manager
        self.progress_callback = None
        self.ffmpeg_path = get_ffmpeg()
        logging.info(f"Initialized Transcriber with FFMPEG path: {self.ffmpeg_path}")
        self.device = self.config.device
        self.faster_whisper_threads = 4
        self.model = None
        
        self.model_path = resource_path("models")
        self.user_chosen_language = None
        if self.config.threads > 0:
            torch.set_num_threads(self.config.threads)
            self.faster_whisper_threads = self.config.threads
        self.min_transcription_length = 100  # Minimum length for reliable language detection
        self.valid_language_codes = set([
            'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar', 'sv', 'it', 'id', 'hi', 
            'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu', 'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 
            'mi', 'ml', 'cy', 'sk', 'te', 'fa', 'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 
            'ne', 'mn', 'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc', 'ka', 'be', 
            'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn', 'mt', 'sa', 'lb', 'my', 'bo', 'tl', 
            'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw', 'su', 'yue'
        ])

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def load_asr_model(self):
        with silent_subprocess():
            with gpu_memory_manager():
                if self.model is None:
                    local_model_path = os.path.join(self.config.whisper_download_root, self.config.whisper_model_name)
                    if os.path.exists(local_model_path):
                        logging.info(f"Using local model: {local_model_path}")
                        self.model = load_model(
                            whisper_arch=local_model_path,
                            device=self.config.device,
                            device_index=self.config.device_index,
                            compute_type=self.config.compute_type,
                            language=self.config.language,
                            asr_options=self.get_asr_options(),
                            vad_options={
                                "vad_onset": self.config.vad_onset,
                                "vad_offset": self.config.vad_offset,
                            },
                            task=self.config.task,
                            threads=self.faster_whisper_threads,
                            vad_model_fp=self.config.vad_model_fp,
                        )
                    else:
                        logging.info(f"Local model not found. Attempting to download: {self.config.whisper_model_name}")
                        self.model = load_model(
                            whisper_arch=self.config.whisper_model_name,
                            device=self.config.device,
                            device_index=self.config.device_index,
                            download_root=self.config.whisper_download_root,
                            compute_type=self.config.compute_type,
                            language=self.config.language,
                            asr_options=self.get_asr_options(),
                            vad_options={
                                "vad_onset": self.config.vad_onset,
                                "vad_offset": self.config.vad_offset,
                            },
                            task=self.config.task,
                            threads=self.faster_whisper_threads,
                            vad_model_fp=self.config.vad_model_fp,
                        )
                return self.model

    def unload_asr_model(self):
        if self.model:
            # For FasterWhisperPipeline, we need to handle it differently
            if hasattr(self.model, 'model'):
                # Access the internal torch model
                if hasattr(self.model.model, 'cpu'):
                    self.model.model.cpu()
            
            # Delete the model
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logging.info("Unloaded ASR model and cleared GPU memory")

    def get_asr_options(self):
        temperature = self.config.temperature
        if (increment := self.config.temperature_increment_on_fallback) is not None:
            temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
        else:
            temperature = [temperature]

        return {
            "beam_size": self.config.beam_size,
            "patience": self.config.patience,
            "length_penalty": self.config.length_penalty,
            "temperatures": temperature,
            "compression_ratio_threshold": self.config.compression_ratio_threshold,
            "log_prob_threshold": self.config.logprob_threshold,
            "no_speech_threshold": self.config.no_speech_threshold,
            "condition_on_previous_text": self.config.condition_on_previous_text,
            "initial_prompt": self.config.initial_prompt,
            "suppress_tokens": [int(x) for x in self.config.suppress_tokens.split(",")],
            "suppress_numerals": self.config.suppress_numerals,
        }

    def verify_language(self, whisper_lang, text):
        if len(text) < self.min_transcription_length:
            logging.warning(f"Text too short for verification: {len(text)} characters")
            return whisper_lang

        logging.info(f"Verifying language: {whisper_lang}")
        logging.info(f"Text sample for verification: '{text[:200]}...'")
        try:
            detected_lang = detect(text)
            logging.info(f"Language detected by langdetect: {detected_lang}")
            return detected_lang if detected_lang in self.valid_language_codes else whisper_lang
        except LangDetectException as e:
            logging.error(f"LangDetect error in verify_language: {str(e)}")
            return whisper_lang

    def detect_language(self, clip1, clip2):
        with silent_subprocess():
            # First pass using clip1
            first_pass_lang, clip1_transcription = self.detect_language_sample(clip1)
            
            if first_pass_lang == 'en':
                # Second pass using clip2
                second_pass_lang, clip2_transcription = self.detect_language_sample(clip2)
                detected_language = second_pass_lang if second_pass_lang != 'en' else first_pass_lang
                verify_transcription = clip2_transcription
            else:
                detected_language = first_pass_lang
                verify_transcription = clip1_transcription

            verified_lang = self.verify_language(detected_language, verify_transcription)
            if verified_lang != detected_language:
                detected_language = verified_lang
            
            return detected_language

    def detect_language_sample(self, audio_clip):
        with silent_subprocess():
            model = self.load_asr_model()
            result = model.transcribe(audio_clip, language=None)
            
            detected_language = result.get('language', 'en')
            
            # Extract transcription from segments
            transcription = ""
            for seg in result.get('segments', []):
                transcription += seg.get('text', '') + " "
            transcription = transcription.strip()
            
            logging.info(f"Detected language: {detected_language}")
            logging.info(f"Transcription sample: '{transcription[:200]}...'")
            
            return detected_language, transcription

    def transcribe(self, prepared_audio, detected_language=None):
        with silent_subprocess():
            with gpu_memory_manager():
                self.user_chosen_language = self.config.language

                if detected_language is None and (self.user_chosen_language == "Automatic" or self.user_chosen_language is None):
                    detected_language, model = self.detect_language(prepared_audio)
                    transcription_language = detected_language
                else:
                    transcription_language = detected_language or self.user_chosen_language
                    model = self.load_asr_model()
                
                # Extract the language code from the tuple
                if isinstance(transcription_language, tuple):
                    transcription_language = transcription_language[0]
                
                # Set up the progress stream
                progress_stream = ProgressStream(self.progress_callback)
                original_stdout = sys.stdout
                try:
                    sys.stdout = progress_stream
                except AttributeError:
                    original_stdout = None

                # Prepare the audio as a numpy array
                audio_array = decode_audio(prepared_audio)

                result = model.transcribe(
                    audio_array,
                    language=transcription_language,
                    batch_size=self.config.batch_size,
                    chunk_size=self.config.chunk_size,
                    print_progress=True
                )

                # Clear the progress callback
                self.set_progress_callback(None)
        
                # Restore stdout
                if original_stdout is not None:
                    sys.stdout = original_stdout

                # Unload the model after transcription
                self.unload_asr_model()

                language_info = {
                    'user_chosen': self.user_chosen_language,
                    'detected': detected_language,
                    'transcription': transcription_language
                }

                return [(result, prepared_audio, language_info)]

    def process_progress_output(self, output):
        try:
            lines = output.split('\n')
            for line in lines:
                if line.startswith('Progress:'):
                    try:
                        progress = float(line.split(':')[1].strip().rstrip('%'))
                        if self.progress_callback:
                            self.progress_callback(progress)
                    except (ValueError, IndexError):
                        logging.warning(f"Failed to parse progress from line: {line}")
        except Exception as e:
            logging.error(f"Error processing progress output: {str(e)}")

    def align_transcriptions(self, transcriptions):
        with silent_subprocess():
            with gpu_memory_manager():
                if self.config.no_align:
                    return transcriptions

                align_language = self.config.language or "en"
                aligned_results = []

                for result, prepared_audio, language_info in transcriptions:
                    input_audio = decode_audio(prepared_audio)

                    try:
                        align_model, align_metadata = load_align_model(
                            align_language,
                            self.device,
                            model_name=self.config.align_model,
                            model_dir=self.config.align_model_dir,
                        )
                    except ValueError as e:
                        if "No default align-model" in str(e):
                            logging.warning(f"Alignment model not available for {align_language}. Falling back to English.")
                            align_model, align_metadata = load_align_model(
                                "en",
                                self.device,
                                model_name=self.config.align_model,
                                model_dir=self.config.align_model_dir,
                            )
                        else:
                            raise

                    if align_model is not None and len(result["segments"]) > 0:
                        result = align(
                            result["segments"],
                            align_model,
                            align_metadata,
                            input_audio,
                            self.device,
                            interpolate_method=self.config.interpolate_method,
                            return_char_alignments=self.config.return_char_alignments,
                            print_progress=self.config.print_progress,
                        )

                    aligned_results.append((result, prepared_audio, language_info))

                del align_model
                gc.collect()

                return aligned_results

    def diarize_transcriptions(self, transcriptions):
        with silent_subprocess():
            with gpu_memory_manager():
                if not self.config.diarize:
                    return transcriptions

                diarize_model = DiarizationPipeline(
                    config_path=self.config.pyannote_config_path,
                    model_path=self.model_path,
                    device=self.device
                )
                diarized_results = []

                try:
                    for result, audio_path, language_info in transcriptions:
                        diarize_segments = diarize_model(
                            audio_path,
                            min_speakers=self.config.min_speakers,
                            max_speakers=self.config.max_speakers,
                        )
                        result = assign_word_speakers(diarize_segments, result)
                        diarized_results.append((result, audio_path, language_info))
                finally:
                    del diarize_model
                    gc.collect()

                return diarized_results

    def write_transcriptions(self, transcriptions):
        writer = get_writer(self.config.output_format, self.config.output_dir)
        writer_args = {
            "highlight_words": self.config.highlight_words,
            "max_line_count": self.config.max_line_count,
            "max_line_width": self.config.max_line_width,
        }

        for result, audio_path, language_info in transcriptions:
            result["language"] = language_info['transcription']
            result["user_chosen_language"] = language_info['user_chosen']
            result["detected_language"] = language_info['detected']
            writer(result, audio_path, writer_args)