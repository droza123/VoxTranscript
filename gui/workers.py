# gui/workers.py
import os
import json
import numpy as np
import logging
import requests
import shutil
import torch
import torchaudio
import asyncio
from PyQt6.QtCore import QThread, pyqtSignal
from transcriber import Transcriber
from datetime import datetime
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier
import wave
import gc
from contextlib import contextmanager

from audio import prepare_audio
from sleep_prevention import sleep_preventer
from ollama_integration import OllamaIntegration
from subprocess_context import silent_subprocess
from utils import force_cuda_memory_release, log_gpu_memory_usage

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

class TranscriptionWorker(QThread):
    progress = pyqtSignal(str, str, int, int, bool)
    file_transcribed = pyqtSignal(str, bool, object, str, dict)  
    error = pyqtSignal(str)
    stop_finished = pyqtSignal()

    def __init__(self, config, file, settings_manager):
        super().__init__()
        self.config = config
        self.file = os.path.normpath(file)
        self.settings_manager = settings_manager
        self.transcriber = None
        self.current_stage = 0
        self.total_stages = self.calculate_total_stages()
        self.use_custom_filenames = self.settings_manager.get('use_custom_filenames', False)
        self.filename_prefix = self.settings_manager.get('filename_prefix', '')
        self.filename_suffix = self.settings_manager.get('filename_suffix', '')
        self.is_stopped = False
        self.is_completed = False
        self.in_transcription_stage = False
        self.temp_files = []
        self.ollama = None

    def calculate_total_stages(self):
        stages = 4  # Base stages: preparation, transcription, alignment, and saving
        if self.config.language is None or self.config.language == "Automatic":
            stages += 1  # Add language detection stage
        if self.config.diarize:
            stages += 1  # Add diarization stage
        if self.config.save_voice_recognition and 'jsonl' in self.settings_manager.get('output_formats', []):
            stages += 1  # Add voice recognition stage
        if self.settings_manager.get('auto_summarize', False):
            stages += 1  # Add summarization stage
        return stages

    def clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        logging.info("Cleared GPU memory and ran garbage collection")

    def run(self):
        if self.is_completed:
            logging.info(f"File {self.file} has already been processed. Skipping.")
            return
        
        logging.info(f"Starting transcription for file: {self.file}")
        prepared_audio = None
        clip1 = None
        clip2 = None
        self.current_stage = 0
        detected_language = None

        try:
            sleep_preventer.prevent_sleep()
            
            # Initialize Transcriber
            self.transcriber = Transcriber(self.config, self.settings_manager)

            # Audio preparation stage
            self.current_stage += 1
            self.progress.emit(self.file, "Preparing audio", self.current_stage, self.total_stages, False)
            prepared_audio, clip1, clip2 = self.prepare_audio_for_transcription()
            if prepared_audio is None:
                raise Exception("Failed to prepare audio")
            self.progress.emit(self.file, "Preparing audio", self.current_stage, self.total_stages, True)

            if self.is_stopped:
                self.handle_stop()
                return

            # Language detection stage (if needed)
            if self.config.language is None or self.config.language == "Automatic":
                self.current_stage += 1
                self.progress.emit(self.file, "Detecting language", self.current_stage, self.total_stages, False)
                detected_language = self.transcriber.detect_language(clip1, clip2)
                self.progress.emit(self.file, "Detecting language", self.current_stage, self.total_stages, True)
                
                # Delete the clips immediately after use
                os.remove(clip1)
                os.remove(clip2)
            else:
                detected_language = self.config.language

            # Transcription stage
            self.current_stage += 1
            self.progress.emit(self.file, "Transcribing audio", self.current_stage, self.total_stages, False)
            self.in_transcription_stage = True
            self.transcriber.set_progress_callback(self.transcription_progress_callback)
            results = self.transcriber.transcribe(prepared_audio, detected_language=detected_language)
            self.in_transcription_stage = False
            self.progress.emit(self.file, "Transcribing audio", self.current_stage, self.total_stages, True)
            
            # Unload ASR model
            self.transcriber.unload_asr_model()
            self.clear_gpu_memory()

            if self.is_stopped:
                self.handle_stop()
                return

            # Alignment stage
            self.current_stage += 1
            self.progress.emit(self.file, "Refining timestamps", self.current_stage, self.total_stages, False)
            aligned_results = self.transcriber.align_transcriptions(results)
            self.progress.emit(self.file, "Refining timestamps", self.current_stage, self.total_stages, True)
            
            # Unload alignment model
            self.transcriber.unload_alignment_model()
            self.clear_gpu_memory()
            
            if self.is_stopped:
                self.handle_stop()
                return

            # Diarization stage (if enabled)
            if self.config.diarize:
                self.current_stage += 1
                self.progress.emit(self.file, "Detecting speakers", self.current_stage, self.total_stages, False)
                try:
                    diarized_results = self.transcriber.diarize_transcriptions(aligned_results)
                    final_results, _, language_info = diarized_results[0]
                except Exception as e:
                    logging.error(f"Error during speaker detection: {str(e)}")
                    self.error.emit(f"Error during speaker detection: {str(e)}. Continuing without diarization.")
                    final_results, _, language_info = aligned_results[0]
                self.progress.emit(self.file, "Detecting speakers", self.current_stage, self.total_stages, True)
                self.transcriber.unload_all_models()  # Unload diarization models
                self.clear_gpu_memory()
            else:
                final_results, _, language_info = aligned_results[0]

            # Voice recognition stage (if enabled)
            if self.config.save_voice_recognition and 'jsonl' in self.settings_manager.get('output_formats', []):
                self.current_stage += 1
                self.progress.emit(self.file, "Calculating voice embeddings", self.current_stage, self.total_stages, False)
                transcript, speakers = self.get_transcript_and_speakers(final_results, prepared_audio)
                self.progress.emit(self.file, "Calculating voice embeddings", self.current_stage, self.total_stages, True)
                self.transcriber.unload_speechbrain_model() # Unload speechbrain model
                self.clear_gpu_memory()
            else:
                transcript, speakers = self.get_transcript_and_speakers(final_results, prepared_audio, skip_embeddings=True)

            # Summarization stage (if enabled)
            if self.settings_manager.get('auto_summarize', False):
                self.current_stage += 1
                self.progress.emit(self.file, "Generating summary", self.current_stage, self.total_stages, False)
                summary = self.generate_summary(transcript)
                self.progress.emit(self.file, "Generating summary", self.current_stage, self.total_stages, True)
            else:
                summary = None

            # Saving stage
            self.current_stage += 1
            self.progress.emit(self.file, "Saving transcription files", self.current_stage, self.total_stages, False)
            file_info = self.get_file_info(prepared_audio)
            save_paths = self.save_transcription(self.file, prepared_audio, final_results, language_info, transcript, speakers, summary)
            self.progress.emit(self.file, "Saving transcription files", self.current_stage, self.total_stages, True)

            self.file_transcribed.emit(self.file, True, save_paths, "Completed", file_info)
            self.is_completed = True
            self.clear_gpu_memory()

        except Exception as e:
            logging.error(f"Error transcribing {self.file}: {str(e)}", exc_info=True)
            self.error.emit(f"Error transcribing {self.file}: {str(e)}")
            self.file_transcribed.emit(self.file, False, {}, "Failed", {})  # Emit Failed status
        finally:
            self.cleanup(prepared_audio, clip1, clip2)
            force_cuda_memory_release()
            log_gpu_memory_usage()

    def transcription_progress_callback(self, progress):
        if self.in_transcription_stage:
            logging.debug(f"Transcription progress: {progress}")
            self.progress.emit(self.file, f"Transcribing audio: {progress}", self.current_stage, self.total_stages, False)

    def detect_language(self, prepared_audio):
        return self.transcriber.detect_language(prepared_audio)
        
    def prepare_audio_for_transcription(self):
        with silent_subprocess():
            logging.info(f"Preparing audio for transcription: {self.file}")
            
            output_dir = self.get_output_directory()
            logging.info(f"Using output directory for prepared audio: {output_dir}")
            
            try:
                prepared_audio = prepare_audio(self.file, output_dir)
                if prepared_audio is None:
                    raise Exception("Failed to prepare audio")
                
                if self.config.language is None or self.config.language == "Automatic":
                    clip1 = self.prepare_language_detection_clip(prepared_audio, 1/3)
                    clip2 = self.prepare_language_detection_clip(prepared_audio, 2/3)
                    return prepared_audio, clip1, clip2
                
                return prepared_audio, None, None
            except Exception as e:
                logging.error(f"Error preparing audio for transcription: {str(e)}", exc_info=True)
                raise

    def prepare_language_detection_clip(self, prepared_audio, start_ratio):
        with silent_subprocess():
            audio, sr = torchaudio.load(prepared_audio)
            duration = audio.shape[1] / sr
            start_time = int(duration * start_ratio)
            end_time = start_time + 30  # 30 seconds clip
            
            clip = audio[:, int(start_time * sr):int(end_time * sr)]
            
            clip_path = f"{os.path.splitext(prepared_audio)[0]}_clip_{start_ratio:.2f}.wav"
            torchaudio.save(clip_path, clip, sr)
            
            return clip_path

    def handle_stop(self):
        logging.info("Transcription stopped")
        self.file_transcribed.emit(self.file, False, "", "Stopped", {})
        self.stop_finished.emit()

    def cleanup(self, prepared_audio=None, clip1=None, clip2=None):
        sleep_preventer.allow_sleep()
        self.delete_model()
        self.unload_transcriber()
        if hasattr(self, 'embedding_model'):
            del self.embedding_model
        self.clear_gpu_memory()
        self.cleanup_temp_files(prepared_audio, clip1, clip2)
        logging.info(f"Transcription process finished for file: {self.file}")
    
    def delete_model(self):
        if self.transcriber and self.transcriber.model:
            del self.transcriber.model
            self.transcriber.model = None
            logging.info("Deleted transcriber model")
        self.clear_gpu_memory()

    def unload_transcriber(self):
        if self.transcriber:
            try:
                self.transcriber.unload_asr_model()
            except Exception as e:
                logging.error(f"Error unloading ASR model: {str(e)}")
            finally:
                del self.transcriber
                self.transcriber = None
        self.clear_gpu_memory()

    def cleanup_temp_files(self, prepared_audio=None, clip1=None, clip2=None):
        files_to_clean = self.temp_files + [prepared_audio, clip1, clip2]
        for file in files_to_clean:
            if file and os.path.exists(file):
                try:
                    os.remove(file)
                    logging.info(f"Removed temporary file: {file}")
                except Exception as e:
                    logging.error(f"Error removing temporary file {file}: {str(e)}")

        self.temp_files = []
    
    def get_output_directory(self):
        if self.settings_manager.get('save_to_media_folder', True):
            return os.path.dirname(self.file)
        else:
            output_dir = self.settings_manager.get('output_folder', '')
            return output_dir if output_dir else os.path.dirname(self.file)
        
    def stop(self):
        self.is_stopped = True
        self.cleanup()
        self.stop_finished.emit()

    def get_transcript_and_speakers(self, result, prepared_audio, skip_embeddings=False):
        speaker_labels = self.generate_speaker_labels(result['segments'])

        if not skip_embeddings:
            try:
                waveform, sample_rate = torchaudio.load(prepared_audio)
                embedding_model = self.initialize_embedding_model()
                transcript, speakers = self.process_segments_with_embeddings(result, speaker_labels, waveform, sample_rate, embedding_model)
            except Exception as e:
                logging.error(f"Error loading prepared audio file or processing embeddings: {str(e)}")
                logging.info("Falling back to processing without embeddings")
                transcript, speakers = self.process_segments_without_embeddings(result, speaker_labels)
            finally:
                if 'embedding_model' in locals():
                    del embedding_model
                torch.cuda.empty_cache()
        else:
            transcript, speakers = self.process_segments_without_embeddings(result, speaker_labels)

        return transcript, speakers

    def process_segments_without_embeddings(self, result, speaker_labels):
        transcript = []
        speakers = {}
        for segment in result['segments']:
            transcript_item, speaker, speaker_data = self.process_segment(segment, speaker_labels, skip_embeddings=True)
            transcript.append(transcript_item)
            if speaker not in speakers:
                speakers[speaker] = speaker_data
            else:
                speakers[speaker]["total_duration"] += speaker_data["total_duration"]
        return transcript, speakers

    def process_segment(self, segment, speaker_labels, skip_embeddings=False):
        original_speaker = segment.get('speaker', 'UNIDENTIFIED')
        speaker = speaker_labels.get(original_speaker, "UNIDENTIFIED")
        text = segment['text'].strip()
        start_time = self.format_timestamp(segment['start'])
        end_time = self.format_timestamp(segment['end'])

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

    def process_segments_with_embeddings(self, result, speaker_labels, waveform, sample_rate, embedding_model):
        transcript = []
        speakers = {}
        min_segment_length = 3.0
        total_segments = len(result['segments'])

        for i, segment in enumerate(result['segments']):
            transcript_item, speaker, speaker_data = self.process_segment(segment, speaker_labels)
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

            # Update progress
            self.progress.emit(self.file, f"Processing voice embeddings: {(i+1)/total_segments*100:.2f}%", self.total_stages - 2, self.total_stages, False)

        for speaker, data in speakers.items():
            if data["embeddings"]:
                avg_embedding = np.mean(data["embeddings"], axis=0)
                speakers[speaker]["average_embedding"] = avg_embedding.tolist()
            else:
                speakers[speaker]["average_embedding"] = []
            del speakers[speaker]["embeddings"]

        return transcript, speakers

    def initialize_embedding_model(self):
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
                self.download_file(url, file_path)
        
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


    def download_file(self, url, file_path):
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

    def generate_summary(self, transcript):
        # Initialize Ollama here, just before we need it
        if not self.ollama:
            chosen_model = self.settings_manager.get('ollama_model', 'llama3.1:70b')
            self.ollama = OllamaIntegration(model=chosen_model)
            logging.info(f"Initialized OllamaIntegration with model: {chosen_model}")

        # Create a clean, readable version of the transcript
        clean_transcript = self.create_clean_transcript(transcript)
        
        # Check if "Nuestro Padre" is mentioned
        includes_nuestro_padre = self.check_for_nuestro_padre(clean_transcript)
        
        async def generate():
            try:
                return await self.ollama.generate_summary(clean_transcript, includes_nuestro_padre)
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
            self.ollama = None
            # Only close the loop if we created a new one
            if loop != asyncio.get_event_loop():
                loop.close()
    
    def check_for_nuestro_padre(self, transcript):
        lower_transcript = transcript.lower()
        return "nuestro padre" in lower_transcript

    def create_clean_transcript(self, transcript):
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

    def save_transcription(self, audio_file, prepared_audio, result, language_info, transcript, speakers, summary=None):
        output_folder = self.settings_manager.get_output_folder(audio_file)
        output_formats = self.settings_manager.get('output_formats')
        
        filename = self.settings_manager.generate_output_filename(audio_file)
        
        save_paths = {}

        if 'txt' in output_formats:
            txt_path = os.path.normpath(os.path.join(output_folder, f"{filename}.txt"))
            self.save_txt_transcription(txt_path, result, self.get_file_info(prepared_audio), self.get_transcription_info(result, language_info), summary)
            save_paths['txt'] = txt_path

        if 'srt' in output_formats:
            srt_path = os.path.normpath(os.path.join(output_folder, f"{filename}.srt"))
            self.save_srt_transcription(srt_path, result)
            save_paths['srt'] = srt_path

        if 'jsonl' in output_formats:
            jsonl_path = os.path.normpath(os.path.join(output_folder, f"{filename}.jsonl"))
            speaker_mapping = self.generate_speaker_labels(result['segments'])
            self.save_jsonl_transcription(jsonl_path, prepared_audio, result, language_info, transcript, speakers, speaker_mapping, summary)
            save_paths['jsonl'] = jsonl_path

        return save_paths

    def process_transcription_results(self, result, transcript, speakers):
        speaker_labels = self.generate_speaker_labels(result['segments'])
        
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

    def save_txt_transcription(self, save_path, result, file_info, transcription_info, summary=None):
        speaker_labels = self.generate_speaker_labels(result['segments'])
        
        with open(save_path, 'w', encoding='utf-8') as f:
            # Write metadata
            f.write("Transcription Metadata:\n")
            f.write("======================\n")
            f.write(f"Filename: {file_info['filename']}\n")
            f.write(f"Duration: {self.format_duration(file_info['duration_seconds'])}\n")
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
                    start_time = self.format_timestamp(segment['start'])
                    if transcription_info['diarization'] and not single_speaker:
                        f.write(f"{start_time} {speaker}:\n")
                    else:
                        f.write(f"{start_time}\n")
                
                paragraph.append(segment['text'].strip())
                last_end_time = segment['end']
            
            if paragraph:
                f.write(' '.join(paragraph) + '\n')
        
        return save_path

    def format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def save_jsonl_transcription(self, save_path, prepared_audio, result, language_info, transcript, speakers, speaker_mapping, summary=None):
        try:
            file_info = self.get_file_info(prepared_audio)
            transcription_info = self.get_transcription_info(result, language_info)

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

    def save_srt_transcription(self, save_path, result):
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

    def get_file_info(self, prepared_audio):
        original_file_size = os.path.getsize(self.file)  # Use original file for size
        duration = self.get_audio_duration(prepared_audio)  # Use prepared audio for duration
        return {
            "filename": os.path.basename(self.file),
            "path": os.path.normpath(self.file),
            "size_bytes": original_file_size,
            "duration_seconds": duration
        }

    def get_audio_duration(self, prepared_audio):
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

    def get_speaker_info(self, result):
        if not self.config.diarize:
            return {
                "chosen": "Disabled",
                "detected": "N/A"
            }
        else:
            return {
                "chosen": "Automatic" if self.config.min_speakers == 0 else str(self.config.min_speakers),
                "detected": str(len(set(segment.get('speaker', 'Unknown') for segment in result['segments'])))
            }

    def get_transcription_info(self, result, language_info):
        speaker_info = self.get_speaker_info(result)
        llm_model = self.settings_manager.get('ollama_model', 'N/A') if self.settings_manager.get('auto_summarize', False) else 'N/A'
        return {
            "date": datetime.now().isoformat(),
            "model": self.config.whisper_model_name,
            "chosen_language": "Automatic" if language_info['user_chosen'] is None else language_info['user_chosen'],
            "detected_language": language_info['detected'] if language_info['detected'] is not None else "N/A",
            "transcription_language": language_info['transcription'],
            "diarization": self.config.diarize,
            "chosen_speaker_number": speaker_info['chosen'],
            "detected_speaker_number": speaker_info['detected'],
            "summary_model": llm_model
        }
        
    def generate_speaker_labels(self, segments):
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

    def get_content_info(self):
        return {
            "title": "",  # Placeholder for manual input
            "date_recorded": "",  # Placeholder for manual input
            "content_type": "",  # Placeholder for manual input
            "speakers": []  # Placeholder for manual input
        }

    def get_transcript(self, result):
        transcript = []
        current_speaker = None
        current_text = []
        speaker_labels = self.generate_speaker_labels(result['segments'])

        for segment in result['segments']:
            speaker = segment.get('speaker', 'Unknown')
            text = segment['text'].strip()

            if speaker != current_speaker:
                if current_speaker is not None:
                    transcript.append({
                        "speaker": speaker_labels[current_speaker],
                        "text": " ".join(current_text)
                    })
                current_speaker = speaker
                current_text = [text]
            else:
                current_text.append(text)

        if current_text:
            transcript.append({
                "speaker": speaker_labels[current_speaker],
                "text": " ".join(current_text)
            })

        return transcript

    @staticmethod
    def format_timestamp(seconds):
        h, r = divmod(seconds, 3600)
        m, s = divmod(r, 60)
        ms = int((s % 1) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

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