import multiprocessing
import torch
from queue import Empty
from transcriber import run_transcription

class TranscriptionManager:
    def __init__(self):
        self.process = None
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

    def start_transcription(self, config, file, settings_manager):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

        self.stop_event.clear()
        self.process = multiprocessing.Process(
            target=self._run_transcription_process,
            args=(config, file, settings_manager, self.input_queue, self.output_queue, self.stop_event)
        )
        self.process.start()

    def _run_transcription_process(self, config, file, settings_manager, input_queue, output_queue, stop_event):
        # Ensure CUDA is initialized in this process
        if torch.cuda.is_available():
            torch.cuda.init()

        try:
            run_transcription(config, file, settings_manager, output_queue, stop_event)
        except Exception as e:
            output_queue.put(('error', str(e)))
        finally:
            # Ensure all CUDA memory is released
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def get_output(self):
        try:
            return self.output_queue.get_nowait()
        except Empty:
            return None

    def stop_transcription(self):
        if self.process and self.process.is_alive():
            self.stop_event.set()
            self.process.join(timeout=5)  # Wait for up to 5 seconds
            if self.process.is_alive():
                self.process.terminate()

    def is_transcription_running(self):
        return self.process and self.process.is_alive()