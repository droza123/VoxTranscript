import os
import sys
import logging
from sleep_prevention import sleep_preventer

# Add the current directory and the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from PyQt6.QtWidgets import QApplication
from gui.main_window import WhisperGUI
from load_resources import get_ffmpeg
from utils import get_app_file_path

class UTF8StreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging():
    log_file = get_app_file_path('voxtranscript.log', 'log')
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler with UTF-8 encoding
    console_handler = UTF8StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure the root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")
    logging.info(f"Log file location: {log_file}")

def initialize_ffmpeg():
    ffmpeg_path = get_ffmpeg()
    logging.info(f"FFMPEG initialized at: {ffmpeg_path}")
    return ffmpeg_path

def main():
    setup_logging()
    ffmpeg_path = initialize_ffmpeg()
    
    # Set environment variables
    os.environ['FFMPEG_PATH'] = ffmpeg_path
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    app = QApplication(sys.argv)
    window = WhisperGUI()
    window.show()

    # Ensure sleep is allowed when the application closes
    app.aboutToQuit.connect(sleep_preventer.allow_sleep)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()