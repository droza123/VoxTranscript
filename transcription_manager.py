import multiprocessing
import torch
import logging
from queue import Empty
from transcriber import run_transcription

class TranscriptionManager:
    def __init__(self):
        self.process = None
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()
        self.temp_files = []

    def start_transcription(self, config, file, settings_manager, temp_files):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()

        self.stop_event.clear()
        self.temp_files = temp_files
        self.process = multiprocessing.Process(
            target=self._run_transcription_process,
            args=(config, file, settings_manager, self.input_queue, self.output_queue, self.stop_event, self.temp_files)
        )
        self.process.start()

    def _run_transcription_process(self, config, file, settings_manager, input_queue, output_queue, stop_event, temp_files):
        # Ensure CUDA is initialized in this process
        if torch.cuda.is_available():
            torch.cuda.init()

        try:
            run_transcription(config, file, settings_manager, output_queue, stop_event, temp_files)
        except Exception as e:
            output_queue.put(('error', str(e)))
        finally:
            # Ensure all CUDA memory is released
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def cleanup(self, temp_files):
        logging.info(f"TranscriptionManager: Starting cleanup with temp files: {temp_files}")
        if self.process and self.process.is_alive():
            logging.info("TranscriptionManager: Process is still alive. Terminating.")
            self.process.terminate()
            self.process.join()
        from transcriber import cleanup as transcriber_cleanup
        transcriber_cleanup(*temp_files)
        logging.info("TranscriptionManager: Cleanup completed.")
               
    def get_output(self):
        try:
            output = self.output_queue.get_nowait()
            if output[0] == 'temp_files':
                self.temp_files = output[1]
            return output
        except Empty:
            return None

    def stop_transcription(self):
        logging.info("TranscriptionManager: Closing the file process.")
        self.stop_event.set()
        if self.process and self.process.is_alive():
            logging.info("TranscriptionManager: Waiting for process to finish.")
            self.process.join(timeout=5)
            if self.process.is_alive():
                logging.info("TranscriptionManager: Process did not terminate. Forcing termination.")
                self.process.terminate()
        logging.info("TranscriptionManager: File process closed.")
    
    def stop_transcription_button_pressed(self):
        logging.info("TranscriptionManager: Stopping transcription.")
        self.stop_event.set()
        if self.process and self.process.is_alive():
            if self.process.is_alive():
                logging.info("TranscriptionManager: File process closed.")
                self.process.terminate()
        logging.info("TranscriptionManager: Transcription stopped.")

    def is_transcription_running(self):
        return self.process and self.process.is_alive()