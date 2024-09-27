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
import traceback
import wave
import requests
import shutil
import torch
import torchaudio
import asyncio
import numpy as np
import json
import aiohttp

from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model, WhisperModel
from whisperx.diarize import assign_word_speakers
from whisperx.utils import get_writer
from utils import resource_path
from langdetect import detect, LangDetectException
from contextlib import contextmanager
from datetime import datetime
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier
from queue import Queue, Empty
from multiprocessing import Queue

from utils import force_cuda_memory_release, log_gpu_memory_usage
from audio import prepare_audio
from sleep_prevention import sleep_preventer
from ollama_integration import OllamaIntegration
from subprocess_context import silent_subprocess
from load_resources import get_ffmpeg
from audio import decode_audio
from pyannote_diarization import get_diarization_pipeline


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

def get_log_file_path():
    log_dir = os.path.join(os.path.expanduser('~'), 'VoxTranscript', 'Logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, 'voxtranscript.log')

def setup_logging():
    log_file = get_log_file_path()
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

def run_transcription(config, file, settings_manager, queue, stop_event, temp_files):
    setup_logging()
    log_file_path = get_log_file_path()
    log_folder_path = os.path.dirname(log_file_path)
    transcriber = None
    is_completed = False
    is_stopped = False
    current_stage = 0
    total_stages = calculate_total_stages(config, settings_manager)
    prepared_audio, clip1, clip2 = temp_files
    detected_language = None

    if is_completed:
        logging.info(f"File {file} has already been processed. Skipping.")
        return

    logging.info(f"Starting transcription for file: {file}")

    try:
        sleep_preventer.prevent_sleep()
        
        # Initializing stage. The initial queue.put command is handled by the main_window.py
        current_stage += 1
        transcriber = Transcriber(config, settings_manager)
        queue.put(('progress', file, "Initializing transcriber", current_stage, total_stages, True))
        
        # Audio preparation stage
        current_stage += 1
        queue.put(('progress', file, "Preparing audio", current_stage, total_stages, False))
        try:
            prepared_audio, clip1, clip2 = prepare_audio_for_transcription(file, config, settings_manager)
            if prepared_audio is None:
                raise Exception("Failed to prepare audio")
            queue.put(('progress', file, "Preparing audio", current_stage, total_stages, True))
        except Exception as e:
            logging.error(f"Error in audio preparation for {file}: {str(e)}", exc_info=True)
            queue.put(('file_transcribed', file, False, {}, "Failed", {}, log_file_path, log_folder_path))
            return
        
        # Language detection stage (if needed)
        if config.language is None or config.language == "Automatic":
            current_stage += 1
            queue.put(('progress', file, "Detecting language", current_stage, total_stages, False))
            detected_language = transcriber.detect_language(clip1, clip2)
            queue.put(('progress', file, "Detecting language", current_stage, total_stages, True))
            
            # Delete the clips immediately after use
            os.remove(clip1)
            os.remove(clip2)
        else:
            detected_language = config.language

        # Transcription stage
        current_stage += 1
        queue.put(('progress', file, "Transcribing audio", current_stage, total_stages, False))
        if transcriber and not is_stopped:
            results = transcriber.transcribe(prepared_audio, detected_language=detected_language)
        else:
            raise Exception("Transcriber is not initialized or process was stopped")
        queue.put(('progress', file, "Transcribing audio", current_stage, total_stages, True))

        # Alignment stage
        if not is_stopped:
            current_stage += 1
            queue.put(('progress', file, "Refining timestamps", current_stage, total_stages, False))
            aligned_results = transcriber.align_transcriptions(results)
            queue.put(('progress', file, "Refining timestamps", current_stage, total_stages, True))

        # Diarization stage (if enabled)
        if config.diarize and not is_stopped:
            current_stage += 1
            queue.put(('progress', file, "Detecting speakers", current_stage, total_stages, False))
            try:
                diarized_results = transcriber.diarize_transcriptions(aligned_results)
                final_results, _, language_info = diarized_results[0]
            except Exception as e:
                logging.error(f"Error during speaker detection: {str(e)}")
                queue.put(('error', f"Error during speaker detection: {str(e)}. Continuing without diarization."))
                final_results, _, language_info = aligned_results[0]
            queue.put(('progress', file, "Detecting speakers", current_stage, total_stages, True))
        else:
            final_results, _, language_info = aligned_results[0]

        # Voice recognition stage (if enabled)
        if config.save_voice_recognition and 'jsonl' in settings_manager.get('output_formats', []) and not is_stopped:
            current_stage += 1
            queue.put(('progress', file, "Calculating voice embeddings", current_stage, total_stages, False))
            try:
                transcript, speakers = get_transcript_and_speakers(final_results, prepared_audio)
                queue.put(('progress', file, "Calculating voice embeddings", current_stage, total_stages, True))
            except Exception as e:
                logging.error(f"Error during voice embedding calculation: {str(e)}")
                queue.put(('error', f"Error during voice embedding calculation: {str(e)}. Continuing without voice recognition."))
                transcript, speakers = get_transcript_and_speakers(final_results, prepared_audio, skip_embeddings=True)
        else:
            transcript, speakers = get_transcript_and_speakers(final_results, prepared_audio, skip_embeddings=True)

        # Summarization stage (if enabled)
        if settings_manager.get('auto_summarize', False) and not is_stopped:
            current_stage += 1
            queue.put(('progress', file, "Generating summary", current_stage, total_stages, False))
            summary = generate_summary(transcript, settings_manager)
            queue.put(('progress', file, "Generating summary", current_stage, total_stages, True))
        else:
            summary = None

        # Saving stage
        current_stage += 1
        queue.put(('progress', file, "Saving transcription files", current_stage, total_stages, False))
        file_info = get_file_info(file, prepared_audio)
        save_paths = save_transcription(file, prepared_audio, final_results, language_info, transcript, speakers, settings_manager, config, summary)
        queue.put(('progress', file, "Saving transcription files", current_stage, total_stages, True))

        # File transcribed successfully
        queue.put(('file_transcribed', file, True, save_paths, "Completed", file_info, log_file_path, log_folder_path))
        is_completed = True

    except Exception as e:
        logging.error(f"Error transcribing {file}: {str(e)}", exc_info=True)
        queue.put(('error', f"Error transcribing {file}: {str(e)}"))
        if not stop_event.is_set():
            queue.put(('file_transcribed', file, False, {}, "Failed", {}, log_file_path, log_folder_path))
    finally:
        cleanup(prepared_audio, clip1, clip2)
        force_cuda_memory_release()
        log_gpu_memory_usage()

def calculate_total_stages(config, settings_manager):
    stages = 5  # Base stages: initializing, preparation, transcription, alignment, and saving
    if config.language is None or config.language == "Automatic":
        stages += 1  # Add language detection stage
    if config.diarize:
        stages += 1  # Add diarization stage
    if config.save_voice_recognition and 'jsonl' in settings_manager.get('output_formats', []):
        stages += 1  # Add voice recognition stage
    if settings_manager.get('auto_summarize', False):
        stages += 1  # Add summarization stage
    return stages

async def check_ollama(settings_manager):
    ollama_url = settings_manager.get('ollama_url', 'http://localhost:11434')
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ollama_url}/api/tags") as response:
                return response.status == 200
    except:
        return False
        
def prepare_audio_for_transcription(file, config, settings_manager):
    with silent_subprocess():
        logging.info(f"Preparing audio for transcription: {file}")
        
        output_dir = get_output_directory(file, settings_manager)
        logging.info(f"Using output directory for prepared audio: {output_dir}")
        
        try:
            prepared_audio = prepare_audio(file, output_dir)
            logging.info(f"Successfully prepared audio: {prepared_audio}")

            clip1 = prepare_language_detection_clip(prepared_audio, start_ratio=1/3)
            clip2 = prepare_language_detection_clip(prepared_audio, start_ratio=2/3)

            return prepared_audio, clip1, clip2
        except Exception as e:
            logging.error(f"Error preparing audio for transcription: {str(e)}", exc_info=True)
            return None, None, None

def get_output_directory(file, settings_manager):
    if settings_manager.get('use_custom_output_folder', False):
        custom_folder = settings_manager.get('custom_output_folder', '')
        if custom_folder:
            return custom_folder
    return os.path.dirname(file)

def handle_stop(queue, file, log_file_path, log_folder_path):
    queue.put(('file_transcribed', file, False, {}, "Stopped", {}, log_file_path, log_folder_path))

def transcription_progress_callback(queue, file, *args):
    queue.put(('transcription_progress', file, *args))

def get_transcript_and_speakers(result, prepared_audio, skip_embeddings=False):
    speaker_labels = generate_speaker_labels(result['segments'])

    if not skip_embeddings:
        try:
            waveform, sample_rate = torchaudio.load(prepared_audio)
            embedding_model = initialize_embedding_model()
            transcript, speakers = process_segments_with_embeddings(result, speaker_labels, waveform, sample_rate, embedding_model)
        except Exception as e:
            logging.error(f"Error loading prepared audio file or processing embeddings: {str(e)}")
            logging.info("Falling back to processing without embeddings")
            transcript, speakers = process_segments_without_embeddings(result, speaker_labels)
        finally:
            if 'embedding_model' in locals():
                del embedding_model
            torch.cuda.empty_cache()
    else:
        transcript, speakers = process_segments_without_embeddings(result, speaker_labels)

    return transcript, speakers

def generate_speaker_labels(segments):
    speaker_labels = {'UNIDENTIFIED': 'UNIDENTIFIED'}
    for i, segment in enumerate(segments):
        original_speaker = segment.get('speaker', f'SPEAKER_{i:02d}')
        if original_speaker not in speaker_labels:
            if original_speaker.startswith('SPEAKER_'):
                speaker_labels[original_speaker] = original_speaker
            else:
                new_label = f'SPEAKER_{len(speaker_labels)-1:02d}'  # -1 because UNIDENTIFIED is already in the dict
                speaker_labels[original_speaker] = new_label
    return speaker_labels

def process_segments_without_embeddings(result, speaker_labels):
    transcript = []
    speakers = {}
    for segment in result['segments']:
        transcript_item, speaker, speaker_data = process_segment(segment, speaker_labels, skip_embeddings=True)
        transcript.append(transcript_item)
        if speaker not in speakers:
            speakers[speaker] = speaker_data
        else:
            speakers[speaker]["total_duration"] += speaker_data["total_duration"]
    return transcript, speakers

def process_segment(segment, speaker_labels, skip_embeddings=False):
    original_speaker = segment.get('speaker', 'UNIDENTIFIED')
    speaker = speaker_labels.get(original_speaker, "UNIDENTIFIED")
    text = segment['text'].strip()
    start_time = format_timestamp(segment['start'])
    end_time = format_timestamp(segment['end'])

    transcript_item = {
        "speaker_label": speaker,
        "text": text,
        "start": start_time,
        "end": end_time,
    }

    speaker_data = {
        "name": "",
        "total_duration": segment['end'] - segment['start']
    }

    if skip_embeddings:
        speaker_data["average_embedding"] = []
    else:
        speaker_data["embeddings"] = []

    return transcript_item, speaker, speaker_data

def process_segments_with_embeddings(result, speaker_labels, waveform, sample_rate, embedding_model):
    transcript = []
    speakers = {}
    min_segment_length = 3.0
    total_segments = len(result['segments'])

    for i, segment in enumerate(result['segments']):
        transcript_item, speaker, speaker_data = process_segment(segment, speaker_labels)
        transcript.append(transcript_item)

        if speaker not in speakers:
            speakers[speaker] = speaker_data
        else:
            speakers[speaker]["total_duration"] += speaker_data["total_duration"]

        segment_duration = segment['end'] - segment['start']
        if segment_duration >= min_segment_length:
            start_sample = int(segment['start'] * sample_rate)
            end_sample = int(segment['end'] * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            try:
                with torch.no_grad():
                    embedding = embedding_model.encode_batch(segment_waveform)
                embedding = embedding.squeeze().cpu().numpy()
                if embedding.ndim == 2:
                    embedding = embedding[0]
                speakers[speaker]["embeddings"].append(embedding)
                logging.info(f"Embedding shape for speaker {speaker}: {embedding.shape}")
            except RuntimeError as e:
                logging.warning(f"Error processing segment for speaker {speaker}: {str(e)}")

    for speaker, data in speakers.items():
        if data["embeddings"]:
            avg_embedding = np.mean(data["embeddings"], axis=0)
            speakers[speaker]["average_embedding"] = avg_embedding.tolist()
        else:
            speakers[speaker]["average_embedding"] = []
        del speakers[speaker]["embeddings"]

    return transcript, speakers

def initialize_embedding_model():
    custom_savedir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "speechbrain_models")
    os.makedirs(custom_savedir, exist_ok=True)
    
    model_files = [
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "classifier.ckpt",
        "label_encoder.txt"
    ]
    
    base_url = "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/"
    
    for file in model_files:
        file_path = os.path.join(custom_savedir, file)
        if not os.path.exists(file_path):
            url = base_url + file
            download_file(url, file_path)
    
    # Create an empty custom.py file
    custom_py_path = os.path.join(custom_savedir, "custom.py")
    if not os.path.exists(custom_py_path):
        open(custom_py_path, 'w').close()
    
    # Ensure label_encoder.txt is copied as label_encoder.ckpt
    shutil.copy2(os.path.join(custom_savedir, "label_encoder.txt"),
                os.path.join(custom_savedir, "label_encoder.ckpt"))
    
    return EncoderClassifier.from_hparams(
        source=custom_savedir,
        savedir=custom_savedir,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def download_file(url, file_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as file, tqdm(
            desc=os.path.basename(file_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error while downloading {url}: {str(e)}")
        raise

def generate_summary(transcript, settings_manager):
        # Initialize Ollama here, just before we need it
        chosen_model = settings_manager.get('ollama_model', 'llama3.1:70b')
        ollama = OllamaIntegration(model=chosen_model)
        logging.info(f"Initialized OllamaIntegration with model: {chosen_model}")

        # Create a clean, readable version of the transcript
        clean_transcript = create_clean_transcript(transcript)
        
        # Check if "Nuestro Padre" is mentioned
        includes_nuestro_padre = check_for_nuestro_padre(clean_transcript)
        
        async def generate():
            try:
                return await ollama.generate_summary(clean_transcript, includes_nuestro_padre)
            except Exception as e:
                logging.error(f"Error generating summary: {str(e)}")
                return None

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            summary = loop.run_until_complete(generate())
            return summary
        except Exception as e:
            logging.error(f"Unexpected error in generate_summary: {str(e)}")
            return None
        finally:
            # We don't need to explicitly unload the model anymore
            ollama = None
            # Only close the loop if we created a new one
            if loop != asyncio.get_event_loop():
                loop.close()

def create_clean_transcript(transcript):
    clean_text = "Transcription:\n\n"
        
    current_speaker = None
    paragraph = []
    
    for segment in transcript:
        speaker = segment['speaker_label']
        text = segment['text'].strip()
        
        if speaker != current_speaker or not paragraph:
            if paragraph:
                clean_text += f"{current_speaker}: {' '.join(paragraph)}\n\n"
                paragraph = []
            current_speaker = speaker
        
        paragraph.append(text)
    
    if paragraph:
        clean_text += f"{current_speaker}: {' '.join(paragraph)}\n\n"
    
    return clean_text.strip()

def check_for_nuestro_padre(clean_transcript):
    lower_transcript = clean_transcript.lower()
    return "nuestro padre" in lower_transcript     

def get_file_info(original_file, prepared_audio):
    original_file_size = os.path.getsize(original_file)  # Use original file for size
    duration = get_audio_duration(prepared_audio)  # Use prepared audio for duration
    return {
        "filename": os.path.basename(original_file),
        "path": os.path.normpath(original_file),
        "size_bytes": original_file_size,
        "duration_seconds": duration
    }

def get_audio_duration(prepared_audio):
    logging.info(f"Attempting to get duration for: {prepared_audio}")
    
    if not os.path.exists(prepared_audio):
        logging.error(f"Prepared audio file does not exist: {prepared_audio}")
        return 0

    file_extension = os.path.splitext(prepared_audio)[1].lower()
    if file_extension != '.wav':
        logging.error(f"Prepared audio file is not a WAV file: {prepared_audio}")
        return 0

    try:
        with wave.open(prepared_audio, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
        logging.info(f"Successfully got duration for {prepared_audio}: {duration} seconds")
        return duration
    except wave.Error as e:
        logging.error(f"Wave module error for {prepared_audio}: {str(e)}")
    except EOFError as e:
        logging.error(f"EOF error for {prepared_audio}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error getting duration for {prepared_audio}: {str(e)}", exc_info=True)
    return 0

def save_transcription(audio_file, prepared_audio, result, language_info, transcript, speakers, settings_manager, config, summary=None):
    output_folder = settings_manager.get_output_folder(audio_file)
    output_formats = settings_manager.get('output_formats')
    
    filename = settings_manager.generate_output_filename(audio_file)
    
    save_paths = {}

    if 'txt' in output_formats:
        txt_path = os.path.normpath(os.path.join(output_folder, f"{filename}.txt"))
        save_txt_transcription(txt_path, result, get_file_info(audio_file, prepared_audio), get_transcription_info(result, language_info, config, settings_manager), summary)
        save_paths['txt'] = txt_path

    if 'srt' in output_formats:
        srt_path = os.path.normpath(os.path.join(output_folder, f"{filename}.srt"))
        save_srt_transcription(srt_path, result)
        save_paths['srt'] = srt_path

    if 'jsonl' in output_formats:
        jsonl_path = os.path.normpath(os.path.join(output_folder, f"{filename}.jsonl"))
        speaker_mapping = generate_speaker_labels(result['segments'])
        save_jsonl_transcription(jsonl_path, audio_file, prepared_audio, result, language_info, transcript, speakers, speaker_mapping, config, settings_manager, summary)
        save_paths['jsonl'] = jsonl_path

    return save_paths

def save_txt_transcription(save_path, result, file_info, transcription_info, summary=None):
    speaker_labels = generate_speaker_labels(result['segments'])
    
    with open(save_path, 'w', encoding='utf-8') as f:
        # Write metadata
        f.write("Transcription Metadata:\n")
        f.write("======================\n")
        f.write(f"Filename: {file_info['filename']}\n")
        f.write(f"Duration: {format_duration(file_info['duration_seconds'])}\n")
        f.write(f"Transcription Date: {transcription_info['date']}\n")
        f.write(f"Model: WhisperX {transcription_info['model']}\n")
        f.write(f"Chosen Language: {transcription_info['chosen_language'] if transcription_info['chosen_language'] is not None else 'Automatic'}\n")
        f.write(f"Detected Language: {transcription_info['detected_language'] if transcription_info['detected_language'] is not None else 'N/A'}\n")
        f.write(f"Transcription Language: {transcription_info['transcription_language']}\n")
        if transcription_info['diarization']:
            f.write(f"Speaker Diarization: Enabled (Detected speakers: {transcription_info['detected_speaker_number']})\n")
        else:
            f.write("Speaker Diarization: Disabled\n")
        f.write(f"Summary Model: {transcription_info['summary_model']}\n")
        f.write("\n")

        # Write summary if available
        if summary:
            f.write("Summary:\n")
            f.write("========\n")
            f.write(summary)
            f.write("\n\n")

        f.write("Transcription:\n")
        f.write("==============\n\n")

        # Write transcription content
        current_speaker = None
        paragraph = []
        last_end_time = 0
        
        unique_speakers = set(speaker_labels.values())
        single_speaker = len(unique_speakers) == 1
        
        for segment in result['segments']:
            original_speaker = segment.get('speaker', 'UNIDENTIFIED')
            speaker = speaker_labels.get(original_speaker, 'UNIDENTIFIED')
            time_gap = segment['start'] - last_end_time
            
            if (transcription_info['diarization'] and not single_speaker and speaker != current_speaker) or time_gap > 2 or not paragraph:
                if paragraph:
                    f.write(' '.join(paragraph) + '\n\n')
                    paragraph = []
                current_speaker = speaker
                start_time = format_timestamp(segment['start'])
                if transcription_info['diarization'] and not single_speaker:
                    f.write(f"{start_time} {speaker}:\n")
                else:
                    f.write(f"{start_time}\n")
            
            paragraph.append(segment['text'].strip())
            last_end_time = segment['end']
        
        if paragraph:
            f.write(' '.join(paragraph) + '\n')
    
    return save_path

def save_srt_transcription(save_path, result):
    def format_timecode(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    with open(save_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], start=1):
            start_time = format_timecode(segment['start'])
            end_time = format_timecode(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

    return save_path

def save_jsonl_transcription(save_path, original_file, prepared_audio, result, language_info, transcript, speakers, speaker_mapping, config, settings_manager, summary=None):
    try:
        file_info = get_file_info(original_file, prepared_audio)
        transcription_info = get_transcription_info(result, language_info, config, settings_manager)

        # Update transcript with new speaker labels, with error checking
        updated_transcript = []
        for item in transcript:
            if item is not None and isinstance(item, dict):
                speaker_label = item.get('speaker_label', 'UNIDENTIFIED')
                new_label = speaker_mapping.get(speaker_label, 'UNIDENTIFIED')
                updated_item = {**item, 'speaker_label': new_label}
                updated_transcript.append(updated_item)

        # Update speakers dictionary
        updated_speakers = {}
        for speaker, data in speakers.items():
            if data is not None and isinstance(data, dict):
                new_speaker_label = speaker_mapping.get(speaker, 'UNIDENTIFIED')
                updated_speakers[new_speaker_label] = {k: v for k, v in data.items() if k != 'name'}

        jsonl_data = {
            "language": language_info.get('transcription'),
            "file_info": file_info,
            "transcription_info": transcription_info
        }

        # Add summary before transcript if available
        if summary:
            jsonl_data["summary"] = summary

        # Add transcript and speakers after summary
        jsonl_data["transcript"] = updated_transcript
        jsonl_data["speakers"] = updated_speakers

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(jsonl_data, f, ensure_ascii=False, indent=2, cls=CustomJSONEncoder)
            f.write('\n')  # Add a newline at the end of the file

        logging.info(f"Successfully saved JSONL transcription to: {save_path}")
    except Exception as e:
        logging.error(f"Error saving JSONL transcription: {str(e)}")
        raise  # Re-raise the exception for further handling

def format_duration(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_transcription_info(result, language_info, config, settings_manager):
    speaker_info = get_speaker_info(result, config)
    
    llm_model = settings_manager.get('ollama_model', 'N/A') if settings_manager.get('auto_summarize', False) else 'N/A'
    
    return {
        "date": datetime.now().isoformat(),
        "model": config.whisper_model_name,
        "chosen_language": "Automatic" if language_info['user_chosen'] is None else language_info['user_chosen'],
        "detected_language": language_info['detected'] if language_info['detected'] is not None else "N/A",
        "transcription_language": language_info['transcription'],
        "diarization": config.diarize,
        "chosen_speaker_number": speaker_info['chosen'],
        "detected_speaker_number": speaker_info['detected'],
        "summary_model": llm_model
    }

def get_speaker_info(result, config):
    if not config.diarize:
        return {
            "chosen": "Disabled",
            "detected": "N/A"
        }
    else:
        return {
            "chosen": "Automatic" if config.min_speakers == 0 else str(config.min_speakers),
            "detected": str(len(set(segment.get('speaker', 'Unknown') for segment in result['segments'])))
        }

def cleanup(prepared_audio=None, clip1=None, clip2=None):
    logging.info(f"Transcriber: Starting cleanup process for files: {prepared_audio}, {clip1}, {clip2}")
    sleep_preventer.allow_sleep()
    delete_model()
    clear_gpu_memory()
    cleanup_temp_files(prepared_audio, clip1, clip2)
    logging.info("Transcriber: Cleanup process completed")

def delete_model():
    if 'transcriber' in globals():
        global transcriber
        if hasattr(transcriber, 'model'):
            try:
                del transcriber.model
                transcriber.model = None
                logging.info("Deleted transcriber model")
            except Exception as e:
                logging.error(f"Error deleting transcriber model: {str(e)}")
        transcriber = None
    clear_gpu_memory()

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    logging.info("Cleared GPU memory and ran garbage collection")

def cleanup_temp_files(prepared_audio=None, clip1=None, clip2=None):
    files_to_clean = [prepared_audio, clip1, clip2]
    for file in files_to_clean:
        if file and os.path.exists(file):
            try:
                os.remove(file)
                logging.info(f"Transcriber: Removed temporary file: {file}")
            except Exception as e:
                logging.error(f"Transcriber: Error removing temporary file {file}: {str(e)}")
        else:
            logging.info(f"Transcriber: Temporary file not found or already removed: {file}")

def process_transcription_results(result, transcript, speakers):
    speaker_labels = generate_speaker_labels(result['segments'])
    
    # Create a reverse mapping from new labels to original labels
    reverse_mapping = {v: k for k, v in speaker_labels.items()}
    
    # Update transcript
    updated_transcript = []
    for item in transcript:
        original_label = reverse_mapping.get(item['speaker_label'], item['speaker_label'])
        new_label = speaker_labels.get(original_label, item['speaker_label'])
        updated_transcript.append({**item, 'speaker_label': new_label})

    # Update speakers
    updated_speakers = {}
    for old_label, data in speakers.items():
        original_label = reverse_mapping.get(old_label, old_label)
        new_label = speaker_labels.get(original_label, old_label)
        updated_speakers[new_label] = data

    # Update segments in the original result
    for segment in result['segments']:
        original_speaker = segment.get('speaker', 'SPEAKER_00')
        segment['speaker'] = speaker_labels.get(original_speaker, original_speaker)

    return result, updated_transcript, updated_speakers, speaker_labels

        
def prepare_language_detection_clip(prepared_audio, start_ratio):
    with silent_subprocess():
        audio, sr = torchaudio.load(prepared_audio)
        duration = audio.shape[1] / sr
        start_time = int(duration * start_ratio)
        end_time = start_time + 30  # 30 seconds clip
        
        clip = audio[:, int(start_time * sr):int(end_time * sr)]
        
        clip_path = f"{os.path.splitext(prepared_audio)[0]}_clip_{start_ratio:.2f}.wav"
        torchaudio.save(clip_path, clip, sr)
        
        return clip_path

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)  # Convert any non-serializable objects to strings
        
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

class Transcriber:
    def __init__(self, config: TranscriptionConfig, settings_manager):
        self.config = config
        self.settings_manager = settings_manager
        self.progress_callback = None
        self.ffmpeg_path = get_ffmpeg()
        logging.info(f"Initialized Transcriber with FFMPEG path: {self.ffmpeg_path}")
        self.device = self.get_device()
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

                # Prepare the audio as a numpy array
                audio_array = decode_audio(prepared_audio)

                result = model.transcribe(
                    audio_array,
                    language=transcription_language,
                    batch_size=self.config.batch_size,
                    chunk_size=self.config.chunk_size,
                    print_progress=True
                )

                # Unload the model after transcription
                self.unload_asr_model()

                language_info = {
                    'user_chosen': self.user_chosen_language,
                    'detected': detected_language,
                    'transcription': transcription_language
                }

                return [(result, prepared_audio, language_info)]

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

    def get_device(self):
        if isinstance(self.config.device, str):
            return torch.device(self.config.device)
        return self.config.device

    def diarize_transcriptions(self, transcriptions):
        with silent_subprocess():
            with gpu_memory_manager():
                if not self.config.diarize:
                    return transcriptions

                diarized_results = []

                try:
                    logging.info(f"Diarization started. Device: {self.device}, Type: {type(self.device)}")
                    
                    with get_diarization_pipeline(
                        config_path=self.config.pyannote_config_path,
                        model_path=self.model_path,
                        device=self.device
                    ) as diarize_model:
                        logging.info("Diarization pipeline created successfully")
                        
                        for result, audio_path, language_info in transcriptions:
                            logging.info(f"Processing audio: {audio_path}")
                            diarize_segments = diarize_model(
                                audio_path,
                                min_speakers=self.config.min_speakers,
                                max_speakers=self.config.max_speakers,
                            )
                            result = assign_word_speakers(diarize_segments, result)
                            diarized_results.append((result, audio_path, language_info))
                            logging.info(f"Diarization completed for {audio_path}")

                except Exception as e:
                    logging.error(f"Error during diarization: {str(e)}")
                    logging.error(f"Error type: {type(e)}")
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    logging.error(f"Device at error: {self.device}, Type: {type(self.device)}")
                    # If diarization fails, return the original transcriptions
                    return transcriptions

                logging.info("Diarization process completed successfully")
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
    
    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logging.info("GPU memory cleared")