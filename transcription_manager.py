import multiprocessing
import torch
import logging
import time
from queue import Empty
from transcriber import run_transcription

class TranscriptionManager:
    def __init__(self):
        self.pool = None
        self.manager = multiprocessing.Manager()
        self.output_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()
        self.current_task = None

    def start_transcription(self, config, file, settings_manager, temp_files):
        logging.info(f"TranscriptionManager: Starting transcription for file: {file}")
        try:
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

            # Check for immediate errors
            time.sleep(1)  # Give the process a moment to start
            if self.current_task.ready():
                try:
                    self.current_task.get(timeout=1)  # This will raise an exception if there was an error
                except Exception as e:
                    logging.error(f"Error occurred immediately after starting transcription: {str(e)}", exc_info=True)
                    self.output_queue.put(('error', f"Error starting transcription: {str(e)}"))
            else:
                logging.info("Transcription task started successfully")

        except Exception as e:
            logging.error(f"Error starting transcription: {str(e)}", exc_info=True)
            self.output_queue.put(('error', f"Error starting transcription: {str(e)}"))

    def get_output(self):
        try:
            output = self.output_queue.get_nowait()
            logging.debug(f"Received output from queue: {output}")
            return output
        except Empty:
            return None
        except Exception as e:
            logging.error(f"Error getting output from queue: {str(e)}", exc_info=True)
            return ('error', f"Error getting output from queue: {str(e)}")

    def stop_transcription(self):
        logging.info("TranscriptionManager: Stopping transcription.")
        self.stop_event.set()
        if self.current_task:
            logging.info("Waiting for current task to complete")
            try:
                self.current_task.get(timeout=5)
                logging.info("Current task completed")
            except multiprocessing.TimeoutError:
                logging.warning("Transcription task did not complete within the timeout period.")
            except Exception as e:
                logging.error(f"Error stopping transcription: {str(e)}", exc_info=True)
        logging.info("TranscriptionManager: Transcription stopped.")

    def stop_transcription_button_pressed(self):
        logging.info("TranscriptionManager: Stop button pressed.")
        self.stop_transcription()

    def is_transcription_running(self):
        is_running = self.current_task is not None and not self.current_task.ready()
        logging.debug(f"TranscriptionManager: Transcription running status: {is_running}")
        return is_running

    def cleanup(self, temp_files):
        logging.info(f"TranscriptionManager: Starting cleanup with temp files: {temp_files}")
        self.stop_transcription()
        if self.pool:
            logging.info("Closing and joining the process pool")
            self.pool.close()
            self.pool.join()
            self.pool = None
        self.manager.shutdown()  # Shutdown the manager
        from transcriber import cleanup as transcriber_cleanup
        transcriber_cleanup(*temp_files)
        logging.info("TranscriptionManager: Cleanup completed.")