import logging
from load_resources import get_ffmpeg
import os
import ffmpeg
import numpy as np
import wave
import sys

from subprocess_context import silent_subprocess

import logging
import os
from load_resources import get_ffmpeg
import ffmpeg

def prepare_audio(file_path, output_dir):
    with silent_subprocess():
        logging.info(f"Preparing audio for: {file_path}")
        logging.info(f"Output directory: {output_dir}")
        
        ffmpeg_path = get_ffmpeg()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_prepared.wav")

        try:
            # Use ffprobe to get input file information
            probe = ffmpeg.probe(file_path)
            
            # Check for audio stream
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            
            if audio_stream is None:
                logging.error(f"No audio stream found in the input file: {file_path}")
                return None

            # Prepare the ffmpeg command
            stream = ffmpeg.input(file_path)
            
            # Apply audio filters
            stream = stream.audio
            stream = stream.filter('aresample', 16000)
            stream = stream.filter('asetrate', 16000)
            
            # Output options
            output = ffmpeg.output(stream, output_file,
                                   acodec='pcm_s16le',
                                   ac=1,
                                   ar='16k',
                                   format='wav')

            # Overwrite output file if it exists
            output = output.overwrite_output()

            # Run the ffmpeg command
            ffmpeg.run(output, cmd=ffmpeg_path, capture_stdout=True, capture_stderr=True)

            if os.path.exists(output_file):
                logging.info(f"Successfully prepared audio: {output_file}")
                return output_file
            else:
                logging.error(f"Failed to create output file: {output_file}")
                return None

        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error preparing audio {file_path}: {e.stderr.decode()}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error preparing audio {file_path}: {str(e)}", exc_info=True)
            return None

def decode_audio(file: str, sampling_rate: int = 16000):
    try:
        with wave.open(file, 'rb') as wav_file:
            # Check if the file matches our expected format
            if (wav_file.getnchannels() != 1 or 
                wav_file.getsampwidth() != 2 or 
                wav_file.getframerate() != sampling_rate):
                raise ValueError("WAV file does not match expected format")

            # Read the entire audio file
            audio_data = wav_file.readframes(wav_file.getnframes())

        # Convert to numpy array
        np_audio = np.frombuffer(audio_data, dtype=np.int16)

        # Normalize to float32 in range [-1.0, 1.0]
        return np_audio.astype(np.float32) / 32768.0

    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {str(e)}") from e

# If you're using faster_whisper, you might want to add this line to override its audio module:
sys.modules['faster_whisper.audio'] = sys.modules[__name__]