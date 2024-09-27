import multiprocessing
import torch
import logging
import time
from queue import Empty
from transcriber import run_transcription

class TranscriptionManager:
    def __init__(self):
        self.pool = None
        self.manager = None
        self.output_queue = None
        self.stop_event = None
        self.current_task = None

    def start_transcription(self, config, file, settings_manager, temp_files):
        logging.info(f"TranscriptionManager: Starting transcription for file: {file}")
        try:
            if self.manager is None:
                self.manager = multiprocessing.Manager()
            if self.output_queue is None:
                self.output_queue = self.manager.Queue()
            if self.stop_event is None:
                self.stop_event = self.manager.Event()

            if self.pool is None:
                logging.info("Creating new process pool")
                self.pool = multiprocessing.Pool(processes=1, maxtasksperchild=1)
                logging.info("Process pool created")

            self.stop_event.clear()
            logging.info("Submitting transcription task to the pool")
            self.current_task = self.pool.apply_async(
                run_transcription,
                (config, file, settings_manager, self.output_queue, self.stop_event, temp_files)
            )
            logging.info("Transcription task submitted to the pool")

        except Exception as e:
            logging.error(f"Error starting transcription: {str(e)}", exc_info=True)
            self.output_queue.put(('error', f"Error starting transcription: {str(e)}"))

    def get_output(self):
        try:
            if self.output_queue is None:
                return None
            output = self.output_queue.get_nowait()
            logging.debug(f"Received output from queue: {output}")
            return output
        except Empty:
            return None
        except Exception as e:
            logging.error(f"Error getting output from queue: {str(e)}", exc_info=True)
            return ('error', f"Error getting output from queue: {str(e)}")

    def stop_transcription(self):
        logging.info("TranscriptionManager: Stopping transcription immediately.")
        if self.stop_event:
            self.stop_event.set()
        
        if self.pool:
            logging.info("Terminating the process pool")
            self.pool.terminate()
            self.pool.join()
            self.pool = None

        self.cleanup()

    def stop_transcription_button_pressed(self):
        logging.info("TranscriptionManager: Stop button pressed.")
        self.stop_transcription()

    def is_transcription_running(self):
        is_running = self.pool is not None
        logging.debug(f"TranscriptionManager: Transcription running status: {is_running}")
        return is_running

    def cleanup(self):
        logging.info("TranscriptionManager: Starting cleanup")
        if self.pool:
            logging.info("Ensuring process pool is terminated")
            self.pool.terminate()
            self.pool.join()
            self.pool = None

        if self.output_queue:
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except:
                    pass
            self.output_queue = None

        if self.stop_event:
            self.stop_event = None

        if self.manager:
            self.manager.shutdown()
            self.manager = None

        self.current_task = None
        logging.info("TranscriptionManager: Cleanup completed.")