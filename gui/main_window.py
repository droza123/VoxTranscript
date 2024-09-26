# gui/main_window.py

import os
import logging
from PyQt6.QtWidgets import QMainWindow, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFrame, QApplication
from PyQt6.QtGui import QIcon, QDesktopServices
from PyQt6.QtCore import Qt, QUrl, QTimer, QMetaObject, Q_ARG
import importlib
import gui.components
importlib.reload(gui.components)
from gui.components import ControlPanel, FileQueueComponent
from transcription_manager import TranscriptionManager
from gui.settings import SettingsManager
from gui.dialogs import IntegratedFileDialog
from collections import OrderedDict
from transcriber import TranscriptionConfig
import torch

from config import PYANNOTE_CONFIG_PATH, ALIGN_MODEL_DIR, VAD_MODEL_FP, LANGUAGE_MAP, FASTER_WHISPER_PATH
from utils import resource_path

class WhisperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.info("Initializing WhisperGUI")
        
        self.setWindowTitle("VoxTranscipt")
        self.setGeometry(100, 100, 1200, 700)
        
        # Set the window icon
        icon_path = resource_path(os.path.join("icons", "voxtranscript-icon.svg"))
        self.setWindowIcon(QIcon(icon_path))
        
        self.file_queue = OrderedDict()
        self.worker = None
        self.is_stopping = False
        self.settings_manager = SettingsManager()
        logging.info(f"Loaded settings: {self.settings_manager.settings}")
        
        # Create file_queue_component first
        self.file_queue_component = FileQueueComponent(self.remove_file_from_queue, self.open_transcription, self.settings_manager)
        
        self.setup_ui()
        self.load_ui_settings()
        self.control_panel.verify_loaded_settings()
        self.apply_styles()
        # Connect FileQueueComponent signals to ControlPanel methods
        self.file_queue_component.files_added.connect(self.control_panel.enable_start_button)
        self.file_queue_component.files_removed.connect(self.control_panel.disable_start_button)
        
        self.transcription_manager = TranscriptionManager()
        self.output_timer = QTimer()
        self.output_timer.timeout.connect(self.check_transcription_output)
        self.output_timer.start(100)  # Check every 100ms

        logging.info("WhisperGUI initialization complete")

        self.showMaximized()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left column (control panel)
        self.control_panel = ControlPanel(
            self.start_transcription,
            self.on_stop,
            self.clear_queue,
            self.settings_manager,
            self.check_existing_transcriptions,
            self.file_queue
        )
        self.control_panel.setMinimumWidth(350)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel)

        # Vertical line separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        main_layout.addWidget(line)

        # Right column (file queue and progress)
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        self.file_queue_component = FileQueueComponent(self.remove_file_from_queue, self.open_transcription, self.settings_manager)
        right_layout.addWidget(self.file_queue_component)
        main_layout.addWidget(right_column, 2)

        # Connect signals
        self.control_panel.files_selected.connect(self.process_selected_files)
        self.control_panel.folder_selected.connect(self.add_folder_to_queue)

        self.setMinimumSize(1000, 600)

        logging.debug("UI setup complete")

    def setup_logging(self):
        log_file = 'whisper_gui.log'
        try:
            logging.basicConfig(
                filename=log_file,
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                filemode='w'  # This will overwrite the log file each time the app starts
            )
            logging.info("Logging initialized")
        except Exception as e:
            logging.info(f"Failed to set up logging: {e}")
            # If we can't log to a file, log to console
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            logging.warning(f"Logging to file failed. Logging to console instead. Error: {e}")
    
    def process_selected_files(self, files, is_folder_selection):
        logging.info(f"Processing {'folder' if is_folder_selection else 'files'}: {files}")
        
        # Ensure the cursor is normal before processing
        while QApplication.overrideCursor():
            QApplication.restoreOverrideCursor()

        # Normalize all file paths
        normalized_files = [os.path.normpath(f) for f in files]
        normalized_queue = {os.path.normpath(f): v for f, v in self.file_queue.items()}
        
        # Filter out files already in the queue
        new_files = [f for f in normalized_files if f not in normalized_queue]
        already_queued = [f for f in normalized_files if f in normalized_queue]
        
        logging.info(f"New files: {len(new_files)}, Already queued: {len(already_queued)}")
        
        # Check for existing transcriptions only among the new files
        existing_transcriptions = self.check_existing_transcriptions(new_files)
        
        logging.info(f"Files with existing transcriptions: {len(existing_transcriptions)}")
        
        files_to_add = new_files.copy()
        
        if existing_transcriptions:
            dialog = IntegratedFileDialog(len(existing_transcriptions), self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # User chose to redo transcriptions, so we keep all files
                logging.info("User chose to redo transcriptions")
            else:
                # User chose not to redo transcriptions, so we remove files with existing transcriptions
                files_to_add = [f for f in new_files if f not in existing_transcriptions]
                logging.info(f"User chose not to redo transcriptions. Files to add: {len(files_to_add)}")
        
        # Add files to queue
        for file in files_to_add:
            self.add_file_to_queue(file)
        
        logging.info(f"Added {len(files_to_add)} files to the queue")
        
        # Update button states
        self.control_panel.set_clear_queue_stop_button_enabled(len(self.file_queue) > 0)
        self.control_panel.start_button.setEnabled(len(self.file_queue) > 0)
        
        # Prepare and show the final information dialog
        info_message = []
        if already_queued:
            info_message.append(f"{len(already_queued)} file(s) were already in the queue and were skipped.")
        
        if files_to_add:
            info_message.append(f"{len(files_to_add)} new file(s) were added to the queue.")
        
        if len(files_to_add) < len(new_files):
            skipped_count = len(new_files) - len(files_to_add)
            info_message.append(f"{skipped_count} file(s) with existing transcriptions were skipped.")
        
        if not info_message:
            info_message.append("No new files were added to the queue.")
        
        QMessageBox.information(self, "Files Processed", "\n\n".join(info_message))
        
        # Update settings
        if files:
            if is_folder_selection:
                self.settings_manager.set('last_folder_directory', os.path.dirname(files[0]))
            else:
                self.settings_manager.set('last_file_directory', os.path.dirname(files[0]))

        logging.info("Finished processing files")

    def check_existing_transcription(self, file):
        output_folder = self.settings_manager.get_output_folder(file)
        output_formats = self.settings_manager.get('output_formats', [])
        output_formats = ['jsonl' if fmt == 'json' else fmt for fmt in output_formats]

        filename = self.settings_manager.generate_output_filename(file)
        
        logging.info(f"Checking existing transcription for file: {file}")
        logging.info(f"Output folder: {output_folder}")
        logging.info(f"Output formats: {output_formats}")
        logging.info(f"Generated filename: {filename}")

        # Check if ALL selected formats exist
        for format in output_formats:
            transcription_path = os.path.normpath(os.path.join(output_folder, f"{filename}.{format}"))
            
            logging.info(f"Checking for transcription file: {transcription_path}")
            if not os.path.exists(transcription_path):
                logging.info(f"Transcription file not found: {transcription_path}")
                return False  # If any format is missing, return False
            else:
                logging.info(f"Transcription file found: {transcription_path}")
        
        logging.info("All transcription files found")
        return True  # Only return True if ALL formats exist

    def check_existing_transcriptions(self, files):
        existing_transcriptions = []
        for file in files:
            if self.check_existing_transcription(file):
                existing_transcriptions.append(file)
        return existing_transcriptions

    def apply_styles(self):
        style_file = resource_path(os.path.join("gui", "styles.qss"))
        with open(style_file, 'r') as f:
            self.setStyleSheet(f.read())

    def load_ui_settings(self):
        logging.info("Loading UI settings")
        self.control_panel.load_settings()

    def save_ui_settings(self):
        logging.info("Saving UI settings")
        self.control_panel.save_settings()

    def update_existing_transcription_check(self):
        for file in list(self.file_queue.keys()):
            if self.check_existing_transcription(file):
                self.remove_file_from_queue(file)
        self.file_queue_component.update_overall_progress()

    def create_transcription_config(self):
        language = self.control_panel.get_selected_language()
        language_code = LANGUAGE_MAP.get(language, None)
        speaker_count = self.control_panel.get_speaker_count()
        return TranscriptionConfig(
            whisper_model_name=self.control_panel.get_selected_model(),
            whisper_download_root=FASTER_WHISPER_PATH,
            device="cuda" if self.control_panel.is_gpu_enabled() and torch.cuda.is_available() else "cpu",
            compute_type="float16" if self.control_panel.is_gpu_enabled() and torch.cuda.is_available() else "int8",
            diarize=self.control_panel.is_diarization_enabled(),
            pyannote_config_path=PYANNOTE_CONFIG_PATH,
            batch_size=4,
            align_model_dir=ALIGN_MODEL_DIR,
            vad_model_fp=VAD_MODEL_FP,
            language=language_code,
            min_speakers=speaker_count,
            max_speakers=speaker_count,
            output_format="txt",
            verbose=True,
            task="transcribe",
            print_progress=True,
            save_voice_recognition=self.control_panel.is_voice_recognition_enabled(),
            auto_summarize=self.control_panel.is_auto_summarize_enabled()
            # ... (any other parameters your TranscriptionConfig expects)
        )
        
    def add_files_to_queue(self, files):
        logging.info(f"Adding files to queue: {files}")
        if files:
            self.settings_manager.set('last_file_directory', os.path.dirname(files[0]))
            logging.info(f"Updated last file directory: {self.settings_manager.get('last_file_directory')}")
            new_files = self.add_new_files(files)
            
            if new_files:
                logging.info(f"Added {len(new_files)} new files to the queue")
                QMessageBox.information(self, "Files Added", f"{len(new_files)} new file(s) added to the queue.")
            else:
                logging.info("No new files added to the queue")

    def add_folder_to_queue(self, folder, include_subfolders):
        logging.info(f"Adding folder to queue: {folder}")
        self.settings_manager.set('last_folder_directory', folder)
        logging.info(f"Updated last folder directory: {folder}")
        
        # Change cursor to wait cursor
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            files_found = self.get_audio_files_from_folder(folder, include_subfolders)
            self.add_files_to_queue(files_found)
        finally:
            # Ensure the cursor is restored if any exception occurs
            QApplication.restoreOverrideCursor()

    def get_audio_files_from_folder(self, folder, include_subfolders):
        audio_files = []
        for root, dirs, files in os.walk(folder):
            if not include_subfolders and root != folder:
                continue
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    full_path = os.path.normpath(os.path.abspath(os.path.join(root, file)))
                    audio_files.append(full_path)
        return audio_files

    def add_new_files(self, files):
        new_files = []
        for file in files:
            file_path = os.path.normpath(os.path.abspath(file))
            if file_path not in self.file_queue:
                self.file_queue[file_path] = None  # Use None as a placeholder value
                self.file_queue_component.add_file_to_queue(file_path)
                new_files.append(file_path)
        return new_files

    def add_file_to_queue(self, file):
        logging.info(f"Adding file to queue: {file}")
        normalized_file = os.path.normpath(file)
        if normalized_file not in self.file_queue:
            self.file_queue[normalized_file] = None  # Use None as a placeholder value
            self.file_queue_component.add_file_to_queue(normalized_file)
            logging.info(f"Added file to queue: {normalized_file}")
            
            # Update the last directory used
            self.settings_manager.set('last_file_directory', os.path.dirname(normalized_file))
            
            # Enable Clear Queue button and Start button
            self.control_panel.set_clear_queue_stop_button_enabled(True)
            self.control_panel.start_button.setEnabled(True)
            
            return True
        else:
            logging.info(f"File already in queue: {normalized_file}")
            return False

    def remove_file_from_queue(self, file):
        logging.info(f"Removing file from queue: {file}")
        normalized_file = os.path.normpath(file)
        if normalized_file in self.file_queue:
            del self.file_queue[normalized_file]
            self.file_queue_component.remove_file_from_queue(normalized_file)
            logging.info(f"Removed file from queue: {normalized_file}")
            
            # Update progress after removal
            self.update_progress_after_removal()
            
            # Update button states
            if not self.file_queue:
                self.control_panel.set_clear_queue_stop_button_enabled(False)
                self.control_panel.start_button.setEnabled(False)
            else:
                self.control_panel.set_clear_queue_stop_button_enabled(True)
                self.control_panel.start_button.setEnabled(True)
            
            return True
        else:
            logging.info(f"File not in queue: {normalized_file}")
            return False

    def update_progress_after_removal(self):
        remaining_files = len(self.file_queue)
        if remaining_files > 0:
            config = self.create_transcription_config()
            stages_per_file = 4 if config.diarize else 3
            if config.save_voice_recognition and 'jsonl' in self.control_panel.get_selected_output_formats():
                stages_per_file += 1  # Add voice recognition stage
            self.file_queue_component.set_total_files(remaining_files, stages_per_file)
            self.file_queue_component.update_overall_progress()
        else:
            self.file_queue_component.reset_progress()

    def clear_queue(self):
        if not self.file_queue:
            return

        reply = QMessageBox.question(self, 'Clear Queue',
                                     "Are you sure you want to clear the entire queue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            for file in list(self.file_queue.keys()):
                self.remove_file_from_queue(file)
            self.file_queue_component.reset_progress()
            self.control_panel.set_clear_queue_stop_button_enabled(False)
            self.control_panel.start_button.setEnabled(False)

    def open_transcription(self, file):
        self.file_queue_component.open_transcription(file)

    def on_stop(self):
        self.transcription_manager.stop_transcription()
        self.set_stopping_state(True)
        self.is_stopping = True

    def set_stopping_state(self, is_stopping):
        if is_stopping:
            self.control_panel.clear_queue_stop_button.setText("Stopping...")
            self.control_panel.set_clear_queue_stop_button_enabled(False)
        else:
            self.set_transcription_running_state(False)
            
            # Enable the button if there are files in the queue
            self.control_panel.set_clear_queue_stop_button_enabled(bool(self.file_queue))
            
            # Enable the Start button
            self.control_panel.start_button.setEnabled(True)
            
            # Reset the stopping flag
            self.is_stopping = False
            
            # Update the status of the stopped file
            current_file = list(self.file_queue.keys())[self.current_file_index]
            self.file_queue_component.reset_file_progress(current_file)
            
            # Show the stop message
            self.show_stop_message()

    def start_transcription(self):
        if not self.file_queue:
            QMessageBox.warning(self, "Warning", "Please select files first.")
            return
        
        # Check if at least one output format is selected
        selected_formats = self.control_panel.get_selected_output_formats()
        if not selected_formats:
            QMessageBox.warning(self, "Warning", "Please select at least one output format.")
            return
        
        # Check if custom output folder exists (if enabled)
        if self.control_panel.is_custom_output_folder_enabled():
            custom_folder = self.control_panel.get_custom_output_folder()
            if not custom_folder:
                QMessageBox.warning(self, "Warning", "Please specify a custom output folder.")
                return
            if not os.path.exists(custom_folder):
                response = QMessageBox.question(self, "Output Folder Not Found",
                                                f"The specified output folder does not exist:\n{custom_folder}\n\nDo you want to create it?",
                                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if response == QMessageBox.StandardButton.Yes:
                    try:
                        os.makedirs(custom_folder)
                    except OSError as e:
                        logging.error(f"Failed to create output folder: {str(e)}")
                        QMessageBox.critical(self, "Error", f"Failed to create output folder: {str(e)}")
                        return
                else:
                    return
        
        self.set_ui_enabled(False)
        self.current_file_index = self.get_next_uncompleted_file_index()
        
        # Reset the queue (change Stopped to Queued and reset progress bars)
        self.file_queue_component.reset_queue()
        
        # Set the transcription running state
        self.control_panel.set_transcription_running(True)
        
        self.start_next_transcription()
        
        # Start the timer when transcription begins
        self.output_timer.start(100)  # Check every 100ms

                
    def start_next_transcription(self):
        if self.current_file_index < len(self.file_queue) and not self.is_stopping:
            config = self.create_transcription_config()
            current_file = list(self.file_queue.keys())[self.current_file_index]
            
            if self.file_queue_component.get_file_status(current_file) == "Completed":
                logging.info(f"File {current_file} has already been processed. Moving to next file.")
                self.current_file_index += 1
                self.start_next_transcription()
                return

            logging.info(f"Starting transcription for file: {current_file}")
            
            self.transcription_manager.start_transcription(config, current_file, self.settings_manager)
        else:
            self.on_transcription_finished()
    
    
    def on_file_transcribed(self, file, success, save_paths, status, file_info, log_file_path, log_folder_path):
        # Use QMetaObject.invokeMethod to ensure this runs on the main thread
        QMetaObject.invokeMethod(self.file_queue_component, "update_file_status",
                                Qt.ConnectionType.QueuedConnection,
                                Q_ARG(str, file),
                                Q_ARG(bool, success),
                                Q_ARG(dict, save_paths),
                                Q_ARG(str, status),
                                Q_ARG(str, log_file_path),
                                Q_ARG(str, log_folder_path))
        
        logging.info(f"File transcribed: {file}, Status: {status}")
        
        # Force cleanup. This does not stop the overall process, just the current file, to cleanup GPU memory
        self.transcription_manager.stop_transcription()
        
        if self.is_stopping:
            self.update_stopped_file_status(file)
        else:
            self.current_file_index += 1
            if self.current_file_index < len(self.file_queue):
                self.start_next_transcription()
            else:
                # All files have been processed, trigger the finished event
                QTimer.singleShot(100, self.on_transcription_finished)
    
    def update_stopped_file_status(self, file):
        self.file_queue_component.update_file_status(file, False, {}, "Stopped", "", "")
        self.file_queue_component.reset_file_progress(file)
    
    def on_worker_finished(self):
        logging.info(f"Worker finished for file index: {self.current_file_index}")
        if self.is_stopping:
            current_file = list(self.file_queue.keys())[self.current_file_index]
            self.file_queue_component.reset_file_progress(current_file)
            self.reset_transcription_state()
        else:
            self.current_file_index += 1
            self.start_next_transcription()
        
        # Clean up the worker
        if self.worker:
            self.worker.stop()
            self.worker = None
        
        # Stop the queue timer
        if hasattr(self, 'queue_timer'):
            self.queue_timer.stop()

    def on_transcription_finished(self):
        self.reset_transcription_state()
        
        # Stop the output checking timer
        self.output_timer.stop()
        
        completed_files = [file for file in self.file_queue 
                        if self.file_queue_component.get_file_status(file) == "Completed"]
        failed_files = [file for file in self.file_queue 
                        if self.file_queue_component.get_file_status(file) == "Failed"]
        incomplete_files = [file for file in self.file_queue 
                            if self.file_queue_component.get_file_status(file) not in ["Completed", "Failed"]]

        if len(completed_files) == len(self.file_queue):
            QMessageBox.information(self, "Transcription Complete", "All files have been processed successfully.")
        else:
            self.generate_summary_report(incomplete_files, failed_files)
            
            message = f"Transcription finished.\n"
            message += f"Completed: {len(completed_files)}\n"
            message += f"Failed: {len(failed_files)}\n"
            message += f"Incomplete: {len(incomplete_files)}\n"
            message += "A summary report has been generated."
            
            msg_box = QMessageBox(QMessageBox.Icon.Warning, "Transcription Results", message, parent=self)
            
            view_report_button = msg_box.addButton("View Report", QMessageBox.ButtonRole.ActionRole)
            open_log_folder_button = msg_box.addButton("Open Log Folder", QMessageBox.ButtonRole.ActionRole)
            ok_button = msg_box.addButton(QMessageBox.StandardButton.Ok)
            
            msg_box.setDefaultButton(ok_button)
            
            while True:
                clicked_button = msg_box.exec()
                
                if msg_box.clickedButton() == view_report_button:
                    self.open_summary_report()
                elif msg_box.clickedButton() == open_log_folder_button:
                    self.open_log_folder()
                else:  # Ok button or close button (X) was clicked
                    break
        
        self.control_panel.set_transcription_running(False)
        
        # Update button states
        self.control_panel.start_button.setEnabled(len(self.file_queue) > 0)
        self.control_panel.set_clear_queue_stop_button_enabled(len(self.file_queue) > 0)
        
        # Clean up the TranscriptionManager
        if self.transcription_manager.is_transcription_running():
            self.transcription_manager.stop_transcription()
        
        # Reset stopping state
        self.is_stopping = False
        
    def check_transcription_output(self):
        output = self.transcription_manager.get_output()
        if output is not None:
            message_type, *args = output
            if message_type == 'progress':
                self.update_progress(*args)
            elif message_type == 'file_transcribed':
                self.on_file_transcribed(*args)
            elif message_type == 'error':
                self.on_error(*args)
        
        # Check if the process has stopped
        if self.is_stopping and not self.transcription_manager.process.is_alive():
            self.set_stopping_state(False)
            self.reset_transcription_state()

    def check_worker_queue(self):
        try:
            while not self.worker_queue.empty():
                message = self.worker_queue.get_nowait()
                if message[0] == 'progress':
                    self.update_progress(*message[1:])
                elif message[0] == 'file_transcribed':
                    self.on_file_transcribed(*message[1:])
                elif message[0] == 'error':
                    self.on_error(message[1])
                elif message[0] == 'ollama_not_running':
                    self.handle_ollama_not_running()
        except Exception as e:
            logging.error(f"Error processing worker queue: {str(e)}")

    def set_transcription_running_state(self, is_running):
        self.control_panel.set_transcription_running(is_running)
        self.control_panel.set_clear_queue_stop_button_enabled(True)
        if is_running:
            self.control_panel.start_button.setEnabled(False)
        else:
            self.control_panel.start_button.setEnabled(bool(self.file_queue))

    def get_next_uncompleted_file_index(self):
        for index, file in enumerate(self.file_queue):
            status = self.file_queue_component.get_file_status(file)
            if status != "Completed":
                return index
        return len(self.file_queue)
    
    def handle_ollama_not_running(self):
        QMessageBox.warning(self, "Ollama Not Running", 
                            "Ollama is not running. Please start Ollama and select a model, or disable the auto-summarize option.")
        self.reset_transcription_state()
        self.control_panel.auto_summarize_toggle.setChecked(False)

    def on_error(self, error_message):
        logging.error(f"Transcription error: {error_message}")
        
        # Get the full traceback
        import traceback
        full_traceback = traceback.format_exc()
        logging.error(f"Full traceback:\n{full_traceback}")
        
        # Show a more detailed error message to the user
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setText("An error occurred during transcription.")
        error_dialog.setInformativeText(error_message)
        error_dialog.setDetailedText(full_traceback)
        error_dialog.setWindowTitle("Transcription Error")
        error_dialog.exec()

        # Handle the error (e.g., move to the next file or stop the process)
        self.handle_transcription_error()
    
    def handle_transcription_error(self):
        # Implement error handling logic here
        # For example, you might want to:
        # 1. Move to the next file in the queue
        # 2. Update the status of the current file
        # 3. If it's a critical error, stop the entire process
        pass

    def update_progress(self, file, status, current_stage, total_stages, stage_complete):
        self.file_queue_component.update_file_progress(file, status, current_stage, total_stages, stage_complete)
        
        # Calculate overall progress
        total_files = len(self.file_queue)
        completed_files = sum(1 for f in self.file_queue if self.file_queue_component.get_file_status(f) == "Completed")
        
        # Only count the current stage if it's complete
        current_file_progress = (current_stage - 1 + int(stage_complete)) / total_stages
        
        overall_progress = min(((completed_files + current_file_progress) / total_files) * 100, 100)
        self.file_queue_component.update_overall_progress(overall_progress)

        # Log progress update only when a stage is complete
        if stage_complete:
            logging.info(f"Overall progress updated: {overall_progress:.2f}%")
            
    def show_stop_message(self):
        completed_count = sum(1 for file in self.file_queue if self.file_queue_component.get_file_status(file) == "Completed")
        total_count = len(self.file_queue)
        remaining_count = total_count - completed_count

        message = f"The transcription process has been stopped.\n\n"
        message += f"Files completed: {completed_count} out of {total_count}\n"
        if remaining_count > 0:
            message += f"There are {remaining_count} file(s) remaining.\n"
            message += "You can press the Start button to resume transcription from where it left off."

        QMessageBox.information(self, "Transcription Stopped", message)
        
        self.reset_transcription_state()
    
    def generate_summary_report(self, incomplete_files, failed_files):
        log_dir = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
        report_path = os.path.join(log_dir, "transcription_summary_report.txt")
        
        with open(report_path, "w") as report:
            report.write("Transcription Summary Report\n")
            report.write("============================\n\n")
            
            report.write(f"Total files processed: {len(self.file_queue)}\n")
            report.write(f"Successfully completed: {len(self.file_queue) - len(incomplete_files)}\n")
            report.write(f"Incomplete: {len(incomplete_files)}\n")
            report.write(f"Failed: {len(failed_files)}\n\n")
            
            if failed_files:
                report.write("Failed Files:\n")
                for file in failed_files:
                    report.write(f"- {file}\n")
                report.write("\n")
            
            if incomplete_files:
                report.write("Incomplete Files:\n")
                for file in incomplete_files:
                    if file not in failed_files:
                        status = self.file_queue_component.get_file_status(file)
                        report.write(f"- {file} (Status: {status})\n")
        
        logging.info(f"Summary report generated: {report_path}")
    
    def open_summary_report(self):
        log_dir = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
        report_path = os.path.join(log_dir, "transcription_summary_report.txt")
        QDesktopServices.openUrl(QUrl.fromLocalFile(report_path))

    def open_log_folder(self):
        log_dir = os.path.dirname(logging.getLoggerClass().root.handlers[0].baseFilename)
        QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir))
        
    def reset_transcription_state(self):
        self.control_panel.start_button.setEnabled(True)
        self.set_transcription_running_state(False)
        self.is_stopping = False
        self.set_ui_enabled(True)
        self.file_queue_component.reset_progress()
        self.current_file_index = self.get_next_uncompleted_file_index()
        
        # Enable the Clear Queue button if there are files in the queue
        self.control_panel.set_clear_queue_stop_button_enabled(bool(self.file_queue))
        logging.info("Transcription state reset")
        
        # Clean up the worker and queue timer
        if self.worker:
            self.worker.stop()
            self.worker = None
        if hasattr(self, 'queue_timer'):
            self.queue_timer.stop()
        
    def set_ui_enabled(self, enabled):
        self.control_panel.set_controls_enabled(enabled)
        self.file_queue_component.set_delete_enabled(enabled)

    def closeEvent(self, event):
        logging.info("Application closing")
        self.save_ui_settings()
        self.control_panel.settings_manager.save_settings()
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        super().closeEvent(event)
