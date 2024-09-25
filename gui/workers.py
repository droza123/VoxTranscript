import os
from PyQt6.QtCore import QThread, pyqtSignal
from transcriber import run_transcription
from queue import Empty

class TranscriptionWorker(QThread):
    progress = pyqtSignal(str, str, int, int, bool)
    file_transcribed = pyqtSignal(str, bool, object, str, dict, str, str)
    error = pyqtSignal(str)
    stop_finished = pyqtSignal()
    ollama_not_running = pyqtSignal()

    def __init__(self, config, file, settings_manager, queue):
        super().__init__()
        self.config = config
        self.file = os.path.normpath(file)
        self.settings_manager = settings_manager
        self.queue = queue
        self.is_stopped = False

    def run(self):
        run_transcription(self.config, self.file, self.settings_manager, self.queue)

        while not self.is_stopped:
            try:
                message = self.queue.get(timeout=0.1)
                self.process_message(message)
            except Empty:
                continue

    def process_message(self, message):
        message_type = message[0]
        if message_type == 'progress':
            self.progress.emit(*message[1:])
        elif message_type == 'file_transcribed':
            self.file_transcribed.emit(*message[1:])
            self.is_stopped = True
        elif message_type == 'error':
            self.error.emit(message[1])
            self.is_stopped = True
        elif message_type == 'ollama_not_running':
            self.ollama_not_running.emit()
            self.is_stopped = True

    def stop(self):
        self.is_stopped = True
        self.stop_finished.emit()