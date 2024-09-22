# gui/components.py
import os
import logging
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox,
    QVBoxLayout, QProgressBar, QSpacerItem, QSizePolicy, QGroupBox,
    QFileDialog, QMessageBox, QScrollArea, QStylePainter, QToolButton,
    QStyle, QStyleOptionComboBox, QListView, QApplication, QFrame,
    QLineEdit
)
from PyQt6.QtCore import Qt, QUrl, QTimer, pyqtSignal, pyqtSlot, QThread, QSize, QRect, QPropertyAnimation, QAbstractAnimation, QEasingCurve
from PyQt6.QtGui import QDesktopServices, QIcon, QFont, QColor, QFontMetrics, QPainter, QPen, QPixmap, QTransform

from ollama_integration import OllamaIntegration
from collections import OrderedDict
from config import FASTER_WHISPER_PATH, LANGUAGE_MAP
from utils import resource_path, check_gpu_availability
from gui.animated_toggle import AnimatedToggle
from gui.dialogs import FolderSelectionDialog

class ControlPanel(QWidget):
    files_selected = pyqtSignal(list, bool)
    folder_selected = pyqtSignal(str, bool)

    def __init__(self, start_callback, stop_callback, clear_queue_callback, settings_manager, check_existing_transcriptions_callback, file_queue):
        super().__init__()
        self.settings_manager = settings_manager
        self.initializing = True
        logging.info("Initializing ControlPanel")
        self.setObjectName("controlPanel")
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.clear_queue_callback = clear_queue_callback
        self.check_existing_transcriptions_callback = check_existing_transcriptions_callback
        self.file_queue = file_queue
        self.output_formats = set(['txt', 'srt', 'jsonl'])

        # Initialize toggle objects as instance variables
        self.detect_speakers_toggle = AnimatedToggle()
        self.voice_recognition_toggle = AnimatedToggle()
        self.save_to_media_toggle = AnimatedToggle()
        self.custom_filename_toggle = AnimatedToggle()
        self.gpu_toggle = AnimatedToggle()
        self.auto_summarize_toggle = AnimatedToggle()

        # Initialize ollama integration
        try:
            self.ollama_integration = OllamaIntegration()
        except Exception as e:
            logging.error(f"Error initializing OllamaIntegration: {str(e)}")
            QMessageBox.warning(self, "Ollama Integration Error", 
                                f"Failed to initialize Ollama integration. Auto-summarization may not work properly.\n\nError: {str(e)}")
            self.ollama_integration = None
            
        self.ollama_model_combo = None

        # Setup UI
        self.setFixedWidth(350)
        self.setup_ui()
        self.connect_signals()
        self.load_settings()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 20, 10, 20)
        main_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        self.add_control_widgets(content_layout)

        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area, 1)

        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 20)
        button_layout.setSpacing(8)

        self.start_button = self.create_button("Start", "play")
        self.clear_queue_stop_button = self.create_button("Clear Queue", "trash-2")

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.clear_queue_stop_button)

        self.start_button.setEnabled(False)
        self.clear_queue_stop_button.setEnabled(False)

        main_layout.addWidget(button_container)

    def add_control_widgets(self, layout):
        # File selection buttons
        self.select_files_button = self.create_button("Select Files", "file-plus")
        self.select_folder_button = self.create_button("Select Folder", "folder-plus")
        layout.addWidget(self.select_files_button)
        layout.addWidget(self.select_folder_button)

        # Model settings
        self.model_combo = self.create_dropdown("Whisper Model", ["large-v2", "medium", "small"], "large-v2")
        self.lang_combo = self.create_dropdown("Language", ["Automatic", "English", "Spanish", "French", "German", "Italian"], "Automatic")
        layout.addWidget(QLabel("Whisper Model:"))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("Language:"))
        layout.addWidget(self.lang_combo)

        # Speaker detection
        detect_speakers_layout = QHBoxLayout()
        detect_speakers_layout.addWidget(QLabel("Detect Speakers"))
        self.detect_speakers_toggle = AnimatedToggle()
        detect_speakers_widget, _ = self.create_toggle("Detect Speakers", self.detect_speakers_toggle)
        layout.addWidget(detect_speakers_widget)

        self.speakers_combo = self.create_dropdown("Number of Speakers", ["Automatic", "2", "3", "4", "5"], "Automatic")
        layout.addWidget(QLabel("Number of Speakers:"))
        layout.addWidget(self.speakers_combo)

        # Advanced settings
        self.advanced_settings = self.create_advanced_settings()
        layout.addWidget(self.advanced_settings)

        layout.addStretch(1)

    def create_advanced_settings(self):
        advanced_box = CollapsibleBox("Advanced Settings")
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Voice Recognition group
        voice_recognition_group = QGroupBox("Voice Recognition")
        voice_recognition_layout = QVBoxLayout(voice_recognition_group)
        voice_recognition_layout.setContentsMargins(10, 10, 10, 10)
        voice_recognition_layout.setSpacing(10)

        # Voice Recognition Data toggle
        voice_recognition_toggle_layout = self.create_toggle_with_info(
            "Save Voice Recognition Data",
            self.voice_recognition_toggle,
            "Saves voice data for future speaker identification.\n"
            "Doesn't improve current speaker detection but\n"
            "extends processing time. Can be disabled if not needed."
        )
        voice_recognition_layout.addLayout(voice_recognition_toggle_layout)

        content_layout.addWidget(voice_recognition_group)

        # LLM Settings group
        llm_group = QGroupBox("LLM Settings")
        llm_layout = QVBoxLayout(llm_group)
        llm_layout.setContentsMargins(10, 10, 10, 10)
        llm_layout.setSpacing(10)

        # Auto-summarization toggle
        auto_summarize_layout = self.create_toggle_with_info(
            "Auto-summarize (using Ollama)",
            self.auto_summarize_toggle,
            "Automatically generates a summary using Ollama\n"
            "and adds it to the beginning of the transcript.\n"
            "Requires Ollama to be installed and running."
        )
        llm_layout.addLayout(auto_summarize_layout)

        # Ollama model selection
        ollama_model_layout = self.create_ollama_model_selection()
        llm_layout.addLayout(ollama_model_layout)

        content_layout.addWidget(llm_group)

        # Output settings group
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(10, 10, 10, 10)
        output_layout.setSpacing(10)

        self.save_to_media_toggle = AnimatedToggle()
        save_to_media_widget, _ = self.create_toggle("Save to media folder", self.save_to_media_toggle)
        output_layout.addWidget(save_to_media_widget)

        output_layout.addWidget(QLabel("Custom output folder:"))
        self.output_folder_widget = QWidget()
        folder_layout = QHBoxLayout(self.output_folder_widget)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(5)
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setReadOnly(True)
        self.output_folder_button = QPushButton("...")
        self.output_folder_button.setFixedWidth(40)
        self.output_folder_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
            }
            QPushButton:hover {
                background-color: #3a80d2;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        folder_layout.addWidget(self.output_folder_edit, 1)
        folder_layout.addWidget(self.output_folder_button, 0)
        output_layout.addWidget(self.output_folder_widget)

        # Custom filename options
        self.custom_filename_toggle = AnimatedToggle()
        custom_filename_widget, _ = self.create_toggle("Use custom filenames", self.custom_filename_toggle)
        output_layout.addWidget(custom_filename_widget)

        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Filename prefix:"))
        self.filename_prefix_input = QLineEdit()
        prefix_layout.addWidget(self.filename_prefix_input)
        output_layout.addLayout(prefix_layout)

        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Filename suffix:"))
        self.filename_suffix_input = QLineEdit()
        suffix_layout.addWidget(self.filename_suffix_input)
        output_layout.addLayout(suffix_layout)

        output_layout.addWidget(QLabel("Output Formats:"))
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setSpacing(10)
        self.txt_checkbox = QCheckBox("TXT")
        self.srt_checkbox = QCheckBox("SRT")
        self.jsonl_checkbox = QCheckBox("JSONL")
        checkbox_layout.addWidget(self.txt_checkbox)
        checkbox_layout.addWidget(self.srt_checkbox)
        checkbox_layout.addWidget(self.jsonl_checkbox)
        output_layout.addLayout(checkbox_layout)

        content_layout.addWidget(output_group)

        # Hardware options group
        hardware_group = QGroupBox("Hardware Options")
        hardware_layout = QVBoxLayout(hardware_group)
        hardware_layout.setContentsMargins(10, 10, 10, 10)
        hardware_layout.setSpacing(10)
        self.gpu_toggle = AnimatedToggle()
        gpu_widget, _ = self.create_toggle("Use GPU (CUDA)", self.gpu_toggle)
        hardware_layout.addWidget(gpu_widget)

        content_layout.addWidget(hardware_group)

        # Set the content layout for the CollapsibleBox and refresh Ollama models
        advanced_box.setContentLayout(content_layout)
        self.refresh_ollama_models()
        
        return advanced_box

    def create_ollama_model_selection(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        label_layout = QHBoxLayout()
        ollama_model_label = QLabel("Ollama Model:")
        
        self.refresh_button = RotatableButton()
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.setFixedSize(24, 24)
        refresh_icon = QIcon(resource_path(os.path.join("icons", "refresh.svg")))
        self.refresh_button.setIcon(refresh_icon)
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-radius: 12px;
            }
        """)
        self.refresh_button.clicked.connect(self.refresh_ollama_models)
        
        label_layout.addWidget(ollama_model_label)
        label_layout.addStretch(1)
        label_layout.addWidget(self.refresh_button)
        
        layout.addLayout(label_layout)
        
        self.ollama_model_combo = ModernComboBox()
        self.ollama_model_combo.setObjectName("ollamaModelCombo")
        self.ollama_model_combo.addItem("Loading models...")
        self.ollama_model_combo.setEnabled(False)
        
        layout.addWidget(self.ollama_model_combo)
        
        # Create animation
        self.refresh_animation = QPropertyAnimation(self.refresh_button, b"rotation")
        self.refresh_animation.setDuration(1000)
        self.refresh_animation.setStartValue(0)
        self.refresh_animation.setEndValue(360)
        self.refresh_animation.setLoopCount(-1)  # Infinite loop
        
        return layout

    def create_toggle_with_info(self, label_text, toggle_instance, tooltip_text):
        layout = QHBoxLayout()
        toggle_widget, _ = self.create_toggle(label_text, toggle_instance)
        
        info_icon = QLabel()
        info_icon.setPixmap(QIcon(resource_path(os.path.join("icons", "info.svg"))).pixmap(16, 16))
        info_icon.setToolTip(tooltip_text)
        info_icon.setToolTipDuration(10000)  # Set tooltip duration to 10 seconds
        info_icon.setMouseTracking(True)
        
        layout.addWidget(toggle_widget)
        layout.addWidget(info_icon)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        return layout

    def refresh_ollama_models(self):
        if self.ollama_integration is None:
            logging.warning("Ollama integration is not available")
            self.ollama_model_combo.clear()
            self.ollama_model_combo.addItem("Ollama integration not available")
            self.ollama_model_combo.setEnabled(False)
            return

        logging.debug("Refreshing Ollama models")
        self.ollama_model_combo.clear()
        self.ollama_model_combo.addItem("Loading models...")
        self.ollama_model_combo.setEnabled(False)
        
        self.refresh_button.start_animation()
        
        self.model_fetcher = OllamaModelFetcher(self.ollama_integration)
        self.model_fetcher.models_fetched.connect(self.update_ollama_models)
        self.model_fetcher.start()

    def update_ollama_models(self, models):
        logging.debug(f"Retrieved models: {models}")
        self.ollama_model_combo.clear()
        if models:
            self.ollama_model_combo.addItems(models)
            self.ollama_model_combo.setEnabled(True)
            default_model = self.settings_manager.get('ollama_model', 'llama3.1:70b')
            index = self.ollama_model_combo.findText(default_model)
            if index >= 0:
                self.ollama_model_combo.setCurrentIndex(index)
            logging.debug(f"Set default model: {default_model}, index: {index}")
        else:
            self.ollama_model_combo.addItem("No models found or Ollama not running")
            self.ollama_model_combo.setEnabled(False)
            logging.debug("No models found or Ollama not running")
        
        self.refresh_button.stop_animation()
        logging.debug(f"Final Ollama model combo items: {[self.ollama_model_combo.itemText(i) for i in range(self.ollama_model_combo.count())]}")

    def rotate_refresh_button(self):
        current_rotation = self.refresh_button.rotation
        new_rotation = (current_rotation + 10) % 360  # Rotate by 10 degrees
        self.refresh_button.rotation = new_rotation

    def fetch_ollama_models(self):
        if self.ollama_integration is None:
            return

        logging.debug("Fetching Ollama models")
        models = self.ollama_integration.get_models()
        logging.debug(f"Retrieved models: {models}")
        self.ollama_model_combo.clear()
        if models:
            self.ollama_model_combo.addItems(models)
            self.ollama_model_combo.setEnabled(True)
            default_model = self.settings_manager.get('ollama_model', 'llama3.1:70b')
            index = self.ollama_model_combo.findText(default_model)
            if index >= 0:
                self.ollama_model_combo.setCurrentIndex(index)
            logging.debug(f"Set default model: {default_model}, index: {index}")
        else:
            self.ollama_model_combo.addItem("No models found or Ollama not running")
            self.ollama_model_combo.setEnabled(False)
            logging.debug("No models found or Ollama not running")
        
        # Stop the rotation after updating the models
        self.refresh_button.stop_animation()

        logging.debug(f"Final Ollama model combo items: {[self.ollama_model_combo.itemText(i) for i in range(self.ollama_model_combo.count())]}")

    def stop_refresh_animation(self):
        self.refresh_animation.stop()
        self.refresh_button.rotation = 0
        self.refresh_button.update()

    def connect_signals(self):
        self.select_files_button.clicked.connect(self.select_files)
        self.select_folder_button.clicked.connect(self.select_folder)
        self.start_button.clicked.connect(self.start_callback)
        self.clear_queue_stop_button.clicked.connect(self.on_clear_queue_stop_button_clicked)
        
        # Connect to on_setting_changed
        self.model_combo.currentIndexChanged.connect(self.on_setting_changed)
        self.lang_combo.currentIndexChanged.connect(self.on_setting_changed)
        self.detect_speakers_toggle.stateChanged.connect(self.on_setting_changed)
        self.speakers_combo.currentIndexChanged.connect(self.on_setting_changed)
        self.voice_recognition_toggle.stateChanged.connect(self.on_setting_changed)
        self.filename_prefix_input.textChanged.connect(self.on_setting_changed)
        self.filename_suffix_input.textChanged.connect(self.on_setting_changed)
        self.gpu_toggle.stateChanged.connect(self.on_setting_changed)

        # Keep separate connections for these
        self.auto_summarize_toggle.stateChanged.connect(self.on_auto_summarize_toggled)
        self.ollama_model_combo.currentIndexChanged.connect(self.on_ollama_model_changed)
        self.custom_filename_toggle.stateChanged.connect(self.on_custom_filename_toggled)
        self.save_to_media_toggle.stateChanged.connect(self.on_save_to_media_toggled)
        self.output_folder_button.clicked.connect(self.select_output_folder)
        self.txt_checkbox.stateChanged.connect(self.on_output_format_changed)
        self.srt_checkbox.stateChanged.connect(self.on_output_format_changed)
        self.jsonl_checkbox.stateChanged.connect(self.on_output_format_changed)

    def on_setting_changed(self):
        if not self.initializing:
            sender = self.sender()
            logging.info(f"Setting changed: {sender.objectName()}")
            self.save_settings()

    def on_auto_summarize_toggled(self, state):
        is_enabled = bool(state)
        self.ollama_model_combo.setEnabled(is_enabled)
        self.settings_manager.set('auto_summarize', is_enabled)
        self.save_settings()

    @pyqtSlot(int)
    def on_ollama_model_changed(self, index):
        selected_model = self.ollama_model_combo.currentText()
        self.settings_manager.set('ollama_model', selected_model)
        self.save_settings()
        # Recreate the OllamaIntegration instance with the new model
        self.ollama_integration = OllamaIntegration(model=selected_model)

    def on_save_to_media_toggled(self, state):
        is_save_to_media = bool(state)
        self.output_folder_edit.setEnabled(not is_save_to_media)
        self.output_folder_button.setEnabled(not is_save_to_media)
        self.settings_manager.set('save_to_media_folder', is_save_to_media)
        self.save_settings()

    def on_custom_filename_toggled(self, state):
        enabled = bool(state)
        self.filename_prefix_input.setEnabled(enabled)
        self.filename_suffix_input.setEnabled(enabled)
        if not self.initializing:
            self.save_settings()

    def save_filename_settings(self):
        logging.info("Saving filename settings")
        self.settings_manager.set('filename_prefix', self.filename_prefix_input.text())
        self.settings_manager.set('filename_suffix', self.filename_suffix_input.text())

    def verify_loaded_settings(self):
        logging.info("Verifying loaded settings...")
        logging.info(f"Model: {self.get_selected_model()}")
        logging.info(f"Language: {self.get_selected_language()}")
        logging.info(f"Diarization: {self.is_diarization_enabled()}")
        logging.info(f"Speaker count: {self.get_speaker_count()}")
        logging.info(f"Voice recognition: {self.is_voice_recognition_enabled()}")
        logging.info(f"Use custom filenames: {self.custom_filename_toggle.isChecked()}")
        logging.info(f"Filename prefix: '{self.filename_prefix_input.text()}'")
        logging.info(f"Filename suffix: '{self.filename_suffix_input.text()}'")
        logging.info(f"Use GPU: {self.is_gpu_enabled()}")
        logging.info(f"Output folder: '{self.output_folder_edit.text()}'")
        logging.info(f"Output formats: {self.output_formats}")

    def update_output_formats(self):
        formats = []
        if self.txt_checkbox.isChecked():
            formats.append('txt')
        if self.srt_checkbox.isChecked():
            formats.append('srt')
        if self.jsonl_checkbox.isChecked():
            formats.append('jsonl')
        self.settings_manager.set('output_formats', formats)

    def create_button(self, text, icon_name):
        button = QPushButton(text)
        icon = QIcon(resource_path(os.path.join("icons", f"{icon_name}.svg")))
        # Create a white version of the icon
        pixmap = icon.pixmap(18, 18)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(255, 255, 255))
        painter.end()
        button.setIcon(QIcon(pixmap))
        button.setIconSize(QSize(18, 18))
        return button

    def create_dropdown(self, label_text, items, recommended_option=None):
        combo = ModernComboBox(recommended_option=recommended_option)
        combo.addItems(items)
        return combo
    
    def create_hardware_options_group(self):
        group = QGroupBox("Hardware Options")
        layout = QVBoxLayout(group)

        # Use GPU toggle
        gpu_widget, self.gpu_toggle_instance = self.create_toggle("Use GPU (CUDA)")
        layout.addWidget(gpu_widget)

        return group

    def create_toggle(self, label_text, toggle_instance=None):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label = QLabel(label_text)
        label.setFont(QFont("Segoe UI", 11))
        
        if toggle_instance is None:
            toggle = AnimatedToggle()
        else:
            toggle = toggle_instance
        
        toggle.setFixedSize(QSize(55, 40))  # Increased size
        
        layout.addWidget(label)
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        layout.addWidget(toggle)
        
        return widget, toggle

    def set_transcription_running(self, is_running):
        if is_running:
            self.clear_queue_stop_button.setText("Stop")
            stop_icon = self.create_button("", "stop-circle").icon()
            self.clear_queue_stop_button.setIcon(stop_icon)
        else:
            self.clear_queue_stop_button.setText("Clear Queue")
            trash_icon = self.create_button("", "trash-2").icon()
            self.clear_queue_stop_button.setIcon(trash_icon)

    def set_clear_queue_stop_button_enabled(self, enabled):
        self.clear_queue_stop_button.setEnabled(enabled)

    def create_checkbox(self, text):
        checkbox = QCheckBox(text)
        checkbox.setFont(QFont("Segoe UI", 14))
        return checkbox

    def select_files(self):
        logging.info("Select Files button clicked")
        last_directory = self.settings_manager.get('last_file_directory', os.path.expanduser('~'))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select audio or video files", last_directory,
            "Media Files (*.wav *.mp3 *.aac *.m4a *.flac *.ogg *.wma *.aiff *.aif "
            "*.mp4 *.avi *.mkv *.mov *.wmv *.webm *.flv);;All Files (*.*)"
        )
        if files:
            normalized_files = [os.path.normpath(f) for f in files]
            new_directory = os.path.normpath(os.path.dirname(normalized_files[0]))
            self.settings_manager.set('last_file_directory', new_directory)
            logging.info(f"Updated last file directory: {new_directory}")
            self.files_selected.emit(normalized_files, False)  # False indicates it's not a folder selection

    def select_folder(self):
        logging.info("Select Folder button clicked")
        last_directory = self.settings_manager.get('last_folder_directory', os.path.expanduser('~'))
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", last_directory)
        if folder:
            normalized_folder = os.path.normpath(folder)
            self.settings_manager.set('last_folder_directory', normalized_folder)
            logging.info(f"Updated last folder directory: {normalized_folder}")
            dialog = FolderSelectionDialog(self)
            if dialog.exec():
                include_subfolders = dialog.include_subfolders()
                logging.info(f"Include subfolders: {include_subfolders}")
                
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    files = self.get_audio_files_from_folder(normalized_folder, include_subfolders)
                finally:
                    QApplication.restoreOverrideCursor()
                
                self.files_selected.emit(files, True)  # True indicates it's a folder selection

    def get_audio_files_from_folder(self, folder, include_subfolders):
        audio_files = []
        supported_extensions = ('.wav', '.mp3', '.aac', '.m4a', '.flac', '.ogg', '.wma', '.aiff', '.aif',
                                '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm', '.flv')
        for root, dirs, files in os.walk(folder):
            if not include_subfolders and root != folder:
                continue
            for file in files:
                if file.lower().endswith(supported_extensions):
                    full_path = os.path.normpath(os.path.join(root, file))
                    audio_files.append(full_path)
        return audio_files

    def process_selected_folder(self, folder, include_subfolders):
        logging.info(f"Processing selected folder: {folder}")
        files = self.get_audio_files_from_folder(folder, include_subfolders)
        self.files_selected.emit(files, True)  # True indicates it's a folder selection

    def load_settings(self):
        logging.info("Loading settings...")
        
        self.set_selected_model(self.settings_manager.get('model', 'large-v2'))
        self.set_selected_language(self.settings_manager.get('language', "Automatic"))
        self.set_diarization(self.settings_manager.get('diarize', True))
        self.set_speaker_count(self.settings_manager.get('speaker_count', "Automatic"))
        self.set_voice_recognition_enabled(self.settings_manager.get('save_voice_recognition', True))
        self.set_auto_summarize(self.settings_manager.get('auto_summarize', False))
        
        # Load Ollama model setting
        ollama_model = self.settings_manager.get('ollama_model', 'llama3.1:70b')
        index = self.ollama_model_combo.findText(ollama_model)
        if index >= 0:
            self.ollama_model_combo.setCurrentIndex(index)
            
        use_gpu = self.settings_manager.get('use_gpu', False)
        self.set_gpu_enabled(use_gpu)
        
        self.load_output_settings()
        
        logging.info("Settings loaded")
        self.initializing = False

    def load_output_settings(self):
        save_to_media = self.settings_manager.get('save_to_media_folder', True)
        self.save_to_media_toggle.setChecked(save_to_media)
        self.output_folder_edit.setText(self.settings_manager.get('output_folder', ''))
        
        self.output_folder_edit.setEnabled(not save_to_media)
        self.output_folder_button.setEnabled(not save_to_media)
        
        use_custom_filenames = self.settings_manager.get('use_custom_filenames', False)
        self.custom_filename_toggle.setChecked(use_custom_filenames)
        self.filename_prefix_input.setEnabled(use_custom_filenames)
        self.filename_suffix_input.setEnabled(use_custom_filenames)
        
        prefix = self.settings_manager.get('filename_prefix', '')
        suffix = self.settings_manager.get('filename_suffix', '')
        logging.info(f"load_output_settings - Retrieved prefix: '{prefix}', suffix: '{suffix}'")
        
        self.filename_prefix_input.setText(prefix)
        self.filename_suffix_input.setText(suffix)

        formats = self.settings_manager.get('output_formats', ['txt', 'srt', 'jsonl'])
        self.output_formats = set(formats)
        self.update_checkbox_states()

    def save_settings(self):
        if not self.initializing:
            logging.info("Saving settings...")
            self.settings_manager.set('model', self.get_selected_model())
            self.settings_manager.set('language', self.get_selected_language())
            self.settings_manager.set('diarize', self.is_diarization_enabled())
            self.settings_manager.set('speaker_count', self.get_speaker_count())
            self.settings_manager.set('save_voice_recognition', self.is_voice_recognition_enabled())
            self.settings_manager.set('auto_summarize', self.is_auto_summarize_enabled())
            self.settings_manager.set('ollama_model', self.ollama_model_combo.currentText())
            self.settings_manager.set('use_custom_filenames', self.custom_filename_toggle.isChecked())
            self.settings_manager.set('filename_prefix', self.filename_prefix_input.text())
            self.settings_manager.set('filename_suffix', self.filename_suffix_input.text())
            self.settings_manager.set('use_gpu', self.is_gpu_enabled())
            self.settings_manager.set('save_to_media_folder', self.save_to_media_toggle.isChecked())
            self.settings_manager.set('output_folder', self.output_folder_edit.text())
            self.settings_manager.set('output_formats', list(self.output_formats))
            logging.info("Settings saved")


    def on_output_format_changed(self):
        sender = self.sender()
        format_type = sender.text().lower()
        
        if sender.isChecked():
            self.output_formats.add(format_type)
        else:
            self.output_formats.discard(format_type)
        
        if not self.output_formats:
            QMessageBox.warning(self, "Format Selection", "At least one output format must be selected.")
            self.output_formats = set(['txt', 'srt', 'jsonl'])
            self.update_checkbox_states()
        
        self.save_settings()

    def get_selected_output_formats(self):
        formats = []
        if self.txt_checkbox.isChecked():
            formats.append('txt')
        if self.srt_checkbox.isChecked():
            formats.append('srt')
        if self.jsonl_checkbox.isChecked():
            formats.append('jsonl')
        return formats

    def set_voice_recognition_enabled(self, enabled):
        self.voice_recognition_toggle.setChecked(enabled)

    def is_voice_recognition_enabled(self):
        return self.voice_recognition_toggle.isChecked()

    def set_auto_summarize(self, enabled):
        self.auto_summarize_toggle.setChecked(enabled)

    def is_auto_summarize_enabled(self):
        return self.auto_summarize_toggle.isChecked()

    def is_custom_output_folder_enabled(self):
        return not self.save_to_media_toggle.isChecked()

    def get_custom_output_folder(self):
        return self.output_folder_edit.text()

    def update_checkbox_states(self):
        self.txt_checkbox.setChecked('txt' in self.output_formats)
        self.srt_checkbox.setChecked('srt' in self.output_formats)
        self.jsonl_checkbox.setChecked('jsonl' in self.output_formats)

    def update_settings_and_emit_change(self):
        self.settings_manager.set('output_formats', list(self.output_formats))

    def set_selected_model(self, model):
        index = self.model_combo.findText(model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def on_clear_queue_stop_button_clicked(self):
        if self.clear_queue_stop_button.text() == "Stop":
            self.stop_callback()
        else:
            self.clear_queue_callback()

    def get_selected_model(self):
        return self.model_combo.currentText().split()[0]  # Get first word

    def set_selected_language(self, language):
        index = self.lang_combo.findText(language)
        if index >= 0:
            self.lang_combo.setCurrentIndex(index)

    def get_selected_language(self):
        return self.lang_combo.currentText().split()[0]  # Get first word

    def set_diarization(self, enabled):
        self.detect_speakers_toggle.setChecked(enabled)

    def is_diarization_enabled(self):
        return self.detect_speakers_toggle.isChecked()

    def set_speaker_count(self, count):
        index = self.speakers_combo.findText(str(count))
        if index >= 0:
            self.speakers_combo.setCurrentIndex(index)

    def get_speaker_count(self):
        if not self.is_diarization_enabled():
            return None
        count = self.speakers_combo.currentText().split()[0]  # Get first word
        return 0 if count == "Automatic" else int(count)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.output_folder_edit.text())
        if folder:
            self.output_folder_edit.setText(folder)
            self.save_settings()

    def set_gpu_enabled(self, enabled):
        gpu_available = check_gpu_availability()
        logging.info(f"GPU available: {gpu_available}")
        self.gpu_toggle.setChecked(enabled and gpu_available)
        self.gpu_toggle.setEnabled(gpu_available)
        logging.info(f"GPU enabled: {enabled and gpu_available}")

    def is_gpu_enabled(self):
        gpu_enabled = self.gpu_toggle.isChecked()
        logging.info(f"GPU enabled status (is_gpu_enabled): {gpu_enabled}")
        return gpu_enabled
    
    def on_auto_summarize_toggled(self, state):
        is_enabled = bool(state)
        self.update_ollama_model_combo_state(is_enabled)
        self.settings_manager.set('auto_summarize', is_enabled)
        self.save_settings()
    
    def update_ollama_model_combo_state(self, is_enabled):
        self.ollama_model_combo.setEnabled(is_enabled)
        self.refresh_button.setEnabled(is_enabled)  # Add this line
        if is_enabled:
            self.ollama_model_combo.setStyleSheet("QComboBox { background-color: white; }")
            self.refresh_button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                    border-radius: 12px;
                }
            """)
        else:
            self.ollama_model_combo.setStyleSheet("QComboBox { background-color: #f0f0f0; }")
            self.refresh_button.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                }
            """)
        self.ollama_model_combo.update()
        self.refresh_button.update()
        
    def set_controls_enabled(self, enabled):
        self.select_files_button.setEnabled(enabled)
        self.select_folder_button.setEnabled(enabled)
        self.set_output_controls_enabled(enabled)

        for combo in [self.model_combo, self.lang_combo, self.speakers_combo]:
            combo.setEnabled(enabled)
        
        self.voice_recognition_toggle.setEnabled(enabled)
        self.detect_speakers_toggle.setEnabled(enabled)
        self.gpu_toggle.setEnabled(enabled)
        self.auto_summarize_toggle.setEnabled(enabled)
        self.update_ollama_model_combo_state(enabled and self.auto_summarize_toggle.isChecked())
        
        self.start_button.setEnabled(enabled)
        self.clear_queue_stop_button.setEnabled(not enabled)
        
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()
    
    def set_output_controls_enabled(self, enabled):
        self.save_to_media_toggle.setEnabled(enabled)
        self.output_folder_widget.setEnabled(enabled)
        self.custom_filename_toggle.setEnabled(enabled)
        self.filename_prefix_input.setEnabled(enabled)
        self.filename_suffix_input.setEnabled(enabled)
        self.txt_checkbox.setEnabled(enabled)
        self.srt_checkbox.setEnabled(enabled)
        self.jsonl_checkbox.setEnabled(enabled)

    def enable_start_button(self):
        self.start_button.setEnabled(True)

    def disable_start_button(self):
        self.start_button.setEnabled(False)


class OllamaModelFetcher(QThread):
    models_fetched = pyqtSignal(list)

    def __init__(self, ollama_integration):
        super().__init__()
        self.ollama_integration = ollama_integration

    def run(self):
        try:
            models = self.ollama_integration.get_models()
            self.models_fetched.emit(models)
        except Exception as e:
            logging.error(f"Error fetching Ollama models: {str(e)}")
            self.models_fetched.emit([])
        
class RotatableButton(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._is_rotating = False

    def start_animation(self):
        if not self._is_rotating:
            self._is_rotating = True
            self._timer.start(50)  # Update every 50ms

    def stop_animation(self):
        self._is_rotating = False
        self._timer.stop()
        self._angle = 0
        self.update()

    def _rotate(self):
        self._angle = (self._angle + 10) % 360  # Increase rotation speed
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self._is_rotating:
            painter.translate(self.width() / 2, self.height() / 2)
            painter.rotate(self._angle)
            painter.translate(-self.width() / 2, -self.height() / 2)
        
        icon = self.icon()
        icon_rect = self.rect().adjusted(2, 2, -2, -2)
        icon.paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter, QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled)

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.is_collapsed = True
        self.animation_duration = 300

        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.content_area = QWidget()
        self.content_area.setMaximumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(self.animation_duration)
        self.animation.finished.connect(self.on_animation_finished)
        
        self.content_area.setMaximumWidth(310)

    @property
    def collapsed(self):
        return self.toggle_button.isChecked()

    def finalize_init(self):
        self.is_initialized = True
        self.content_area.setMaximumHeight(0)

    def setContentLayout(self, layout):
        self.content_area.setLayout(layout)
        self.collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        self.content_height = layout.sizeHint().height()
        self.animation.setStartValue(0)
        self.animation.setEndValue(self.content_height)

    def on_pressed(self):
        self.is_collapsed = not self.is_collapsed
        self.toggle_button.setArrowType(
            Qt.ArrowType.RightArrow if self.is_collapsed else Qt.ArrowType.DownArrow
        )
        self.animation.setDirection(
            QAbstractAnimation.Direction.Forward if not self.is_collapsed else QAbstractAnimation.Direction.Backward
        )
        self.animation.start()

    def on_animation_finished(self):
        if not self.is_collapsed:
            self.content_area.setMaximumHeight(self.content_height)
        else:
            self.content_area.setMaximumHeight(0)

class ModernComboBox(QComboBox):
    def __init__(self, parent=None, recommended_option=None):
        super().__init__(parent)
        self.setIconSize(QSize(24, 24))
        self.chevron_icon = self.create_chevron_icon()
        self.recommended_option = recommended_option
        self.setView(QListView())
        self.view().setTextElideMode(Qt.TextElideMode.ElideNone)
        self._original_items = []
        
        # Set font for both the ComboBox and its view
        font = QFont("Segoe UI", 12)  # Adjust size as needed
        self.setFont(font)
        self.view().setFont(font)

    def create_chevron_icon(self):
        icon = QIcon()
        
        # Create normal icon
        normal_pixmap = QIcon(resource_path(os.path.join("icons", "chevron-down.svg"))).pixmap(24, 24)
        icon.addPixmap(normal_pixmap, QIcon.Mode.Normal, QIcon.State.Off)
        
        # Create disabled icon
        disabled_pixmap = QPixmap(normal_pixmap)
        painter = QPainter(disabled_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(disabled_pixmap.rect(), QColor(128, 128, 128))  # Grey color
        painter.end()
        icon.addPixmap(disabled_pixmap, QIcon.Mode.Disabled, QIcon.State.Off)
        
        return icon

    def addItems(self, items):
        self._original_items = items.copy()
        super().addItems(items)
        self._adjust_items()

    def _adjust_items(self):
        max_width = 0
        fm = QFontMetrics(self.font())
        for i, item_text in enumerate(self._original_items):
            display_text = item_text
            if self.recommended_option and item_text == self.recommended_option:
                display_text += " (Recommended)"
            self.setItemText(i, display_text)
            width = fm.horizontalAdvance(display_text)
            max_width = max(max_width, width)
        
        # Add extra space for the chevron and padding
        max_width += 40
        self.setMinimumWidth(max_width)

    def paintEvent(self, event):
        painter = QStylePainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))

        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)

        # Draw the combobox frame and background
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)

        # Calculate text and icon rectangles
        text_rect = self.style().subElementRect(QStyle.SubElement.SE_ComboBoxFocusRect, opt, self)
        icon_size = self.iconSize()
        icon_rect = QRect(self.width() - icon_size.width() - 8, 
                          text_rect.top() + (text_rect.height() - icon_size.height()) // 2,
                          icon_size.width(), icon_size.height())
        
        # Adjust text rectangle to leave space for the icon
        text_rect.setRight(icon_rect.left() - 4)

        # Draw the text
        text = opt.currentText
        painter.setFont(self.font())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)

        # Draw the chevron icon
        if self.chevron_icon:
            mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
            self.chevron_icon.paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter, mode)

    def showPopup(self):
        super().showPopup()
        self.view().setMinimumWidth(self.width())
        self.update()

    def hidePopup(self):
        super().hidePopup()
        self.update()

    def sizeHint(self):
        hint = super().sizeHint()
        return QSize(max(self.minimumWidth(), hint.width()), hint.height())
    
    def setEnabled(self, enabled):
        super().setEnabled(enabled)
        self.update()  # Force a repaint

class FileCardWidget(QWidget):
    def __init__(self, file, remove_callback, open_callback, settings_manager):
        super().__init__()
        self.setObjectName("fileCard")
        self.file = file
        self.settings_manager = settings_manager
        self.save_paths = {}
        self.total_stages = 4  # Default to 4 stages
        self.current_stage = 0

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(5)

        # Content widget (file info and buttons)
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        # Left side - file info
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.file_label = QLabel(os.path.basename(file))
        self.file_label.setObjectName("fileLabel")
        self.file_label.setWordWrap(True)

        self.status_label = QLabel("Status: Queued")
        self.status_label.setObjectName("statusLabel")

        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.status_label)

        # Right side - buttons
        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.open_file_button = self.create_icon_button("file-text", "Open transcription")
        self.open_folder_button = self.create_icon_button("folder-open", "Open containing folder")
        self.remove_button = self.create_icon_button("trash-2", "Remove from queue")

        self.open_file_button.setEnabled(False)

        right_layout.addWidget(self.open_file_button)
        right_layout.addWidget(self.open_folder_button)
        right_layout.addWidget(self.remove_button)

        # Add left and right widgets to content layout
        content_layout.addWidget(left_widget, 4)
        content_layout.addWidget(right_widget, 1)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("fileCardProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        # Apply custom style sheet directly to the progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #e6e6e6;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 4px;
            }
        """)

        # Add content widget and progress bar to main layout
        main_layout.addWidget(content_widget)
        main_layout.addWidget(self.progress_bar)

        self.open_file_button.clicked.connect(self.open_transcription)
        self.open_folder_button.clicked.connect(self.open_containing_folder)
        self.remove_button.clicked.connect(lambda: remove_callback(file))

        self.setFixedHeight(110)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Determine the border color based on the status
        if hasattr(self, 'status_label') and self.status_label.text() == "Status: Failed":
            border_color = QColor("#e74c3c")  # Red color for failed status
        else:
            border_color = QColor("#e0e0e0")  # Default border color

        # Draw the border
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)
    
    def force_repaint(self):
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def create_icon_button(self, icon_name, tooltip):
        button = QPushButton()
        
        # Create icon with normal and disabled states
        icon = QIcon()
        sizes = [24, 48]  # Add more sizes if needed
        for size in sizes:
            # Normal state
            normal_pixmap = QIcon(resource_path(os.path.join("icons", f"{icon_name}.svg"))).pixmap(size, size)
            icon.addPixmap(normal_pixmap, QIcon.Mode.Normal, QIcon.State.Off)
            
            # Disabled state
            disabled_pixmap = QPixmap(normal_pixmap)
            painter = QPainter(disabled_pixmap)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(disabled_pixmap.rect(), QColor(128, 128, 128))  # Grey color
            painter.end()
            icon.addPixmap(disabled_pixmap, QIcon.Mode.Disabled, QIcon.State.Off)
        
        button.setIcon(icon)
        button.setIconSize(QSize(24, 24))
        button.setToolTip(tooltip)
        button.setFixedSize(QSize(36, 36))
        button.setStyleSheet(r"""
            QPushButton {
                background-color: transparent;
                border: none;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-radius: 18px;
            }
            QPushButton:disabled {
                background-color: transparent;
            }
        """)
        return button

    def open_transcription(self):
        if self.save_paths:
            # Prioritize opening files in this order: txt, srt, json
            for ext in ['txt', 'srt', 'json']:
                if ext in self.save_paths:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(self.save_paths[ext]))
                    break

    def open_containing_folder(self):
        folder_path = os.path.normpath(os.path.dirname(self.file))
        if os.path.exists(folder_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder_path))
        else:
            logging.warning(f"Folder not found: {folder_path}")
            QMessageBox.warning(self, "Folder Not Found", f"The folder {folder_path} does not exist.")

    def update_status(self, status, current_stage, total_stages, stage_complete, save_paths=None):
        self.status_label.setText(f"Status: {status}")
        self.total_stages = total_stages
        self.current_stage = current_stage
        
        if save_paths:
            self.save_paths = save_paths

        if status == "Stopped":
            self.current_stage = 0
            self.progress_bar.setValue(0)
        elif status == "Failed":
            self.progress_bar.setValue(100)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    background-color: #e6e6e6;
                    border: 1px solid #d0d0d0;
                    border-radius: 5px;
                }
                QProgressBar::chunk {
                    background-color: #e74c3c;
                    border-radius: 4px;
                }
            """)
            self.setStyleSheet("""
                #fileCard {
                    background-color: #fadbd8;
                    border: 2px solid #e74c3c;
                }
            """)
        elif stage_complete:
            progress = int((self.current_stage / self.total_stages) * 100)
            self.progress_bar.setValue(progress)

        self.open_file_button.setEnabled(status == "Completed" and bool(self.save_paths))
        self.open_folder_button.setEnabled(status == "Completed" and bool(self.save_paths))
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.repaint()  # Ensure the progress bar updates visually

    def set_open_button_enabled(self, enabled):
        self.open_file_button.setEnabled(enabled)
    
    def set_delete_enabled(self, enabled):
        self.remove_button.setEnabled(enabled)

class FileQueueComponent(QWidget):
    files_added = pyqtSignal()  # Signal to indicate files have been added
    files_removed = pyqtSignal()  # Signal to indicate files have been removed

    def __init__(self, remove_callback, open_callback, settings_manager):
        super().__init__()
        self.settings_manager = settings_manager
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(0)
        self.total_stages = 0
        self.completed_stages = 0
        
        # Queue area
        self.queue_widget = QWidget()
        self.queue_layout = QVBoxLayout(self.queue_widget)
        self.queue_layout.setContentsMargins(0, 0, 0, 0)
        self.queue_layout.setSpacing(15)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)  # Remove frame
        self.scroll_area.setStyleSheet("background-color: transparent;")
        
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scrollContent")
        self.scroll_content.setStyleSheet("background-color: transparent;")
        
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_layout.setSpacing(15)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.queue_layout.addWidget(self.scroll_area)

        # Progress bar component
        self.progress_bar = ProgressBarComponent()
        self.queue_layout.addWidget(self.progress_bar)
        
        # Empty state widget
        self.empty_state_widget = QWidget()
        empty_state_layout = QVBoxLayout(self.empty_state_widget)
        empty_state_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create a horizontal layout for icon and text
        icon_text_layout = QHBoxLayout()
        icon_text_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_text_layout.setSpacing(20)  # Add some space between icon and text

        # Create QLabel for the icon
        icon_label = QLabel()
        icon = QIcon(resource_path(os.path.join("icons", "voxtranscript-icon.svg")))
        pixmap = icon.pixmap(128, 128)
        painter = QPainter(pixmap)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        #painter.fillRect(pixmap.rect(), QColor("#666"))  # Use the same color as the text
        painter.end()
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.empty_state_label = QLabel("No files in queue.\nClick 'Select Files' or 'Select Folder'\nto add audio files for transcription.")
        self.empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_label.setStyleSheet("""
            font-size: 16px;
            color: #666;
            margin: 20px;
        """)

        # Add icon and text to the horizontal layout
        icon_text_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignVCenter)
        icon_text_layout.addWidget(self.empty_state_label, alignment=Qt.AlignmentFlag.AlignVCenter)

        # Add the horizontal layout to the empty state layout
        empty_state_layout.addLayout(icon_text_layout)

        # Add both widgets to main layout
        self.main_layout.addWidget(self.queue_widget)
        self.main_layout.addWidget(self.empty_state_widget)
        
        self.file_cards = OrderedDict()
        self.remove_callback = remove_callback
        self.open_callback = open_callback

        # Set initial visibility
        self.update_empty_state()

    def update_empty_state(self):
        if len(self.file_cards) == 0:
            self.queue_widget.hide()
            self.empty_state_widget.show()
        else:
            self.queue_widget.show()
            self.empty_state_widget.hide()

    def add_files_to_queue(self, files):
        logging.info(f"Adding files to queue: {files}")
        card = FileCardWidget(file, self.remove_callback, self.open_transcription, self.settings_manager)
        self.file_cards[file] = card
        if files:
            self.settings_manager.set('last_file_directory', os.path.dirname(files[0]))
            logging.info(f"Updated last file directory: {self.settings_manager.get('last_file_directory')}")
            new_files = [file for file in files if file not in self.file_queue]
            
            for file in new_files:
                self.file_queue[file] = None  # Use None as a placeholder value
                self.file_queue_component.add_file_to_queue(file)
            
            if new_files:
                logging.info(f"Added {len(new_files)} new files to the queue")
                QMessageBox.information(self, "Files Added", f"{len(new_files)} new file(s) added to the queue.")
            else:
                logging.info("No new files added to the queue")

    def add_file_to_queue(self, file):
        card = FileCardWidget(file, self.remove_callback, self.open_callback, self.settings_manager)
        self.file_cards[file] = card
        self.scroll_layout.addWidget(card)
        card.force_repaint()
        self.update()
        self.update_empty_state()
        self.files_added.emit()  # Emit signal when a file is added

    def update_styles(self):
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def open_transcription(self, file):
        if file in self.file_cards:
            self.file_cards[file].open_transcription()

    def remove_file_from_queue(self, file):
        if file in self.file_cards:
            self.scroll_layout.removeWidget(self.file_cards[file])
            self.file_cards[file].deleteLater()
            del self.file_cards[file]
        self.update_empty_state()
        if not self.file_cards:  # If no files left in the queue
            self.files_removed.emit()  # Emit signal when all files are removed

    def update_file_progress(self, file, status, current_stage, total_stages, stage_complete):
        if file in self.file_cards:
            self.file_cards[file].update_status(status, current_stage, total_stages, stage_complete)
            if stage_complete:
                self.completed_stages += 1
                self.update_overall_progress()
    
    def reset_file_progress(self, file):
        if file in self.file_cards:
            self.completed_stages -= self.file_cards[file].current_stage
            self.file_cards[file].update_status("Stopped", 0, self.file_cards[file].total_stages, True)
            self.file_cards[file].progress_bar.setValue(0)
        self.update_overall_progress()

    def update_file_status(self, file, success, save_paths, status):
        if file in self.file_cards:
            previous_stage = self.file_cards[file].current_stage
            self.file_cards[file].update_status(status, self.file_cards[file].total_stages, self.file_cards[file].total_stages, True, save_paths)
            self.completed_stages += (self.file_cards[file].total_stages - previous_stage)
        self.update_overall_progress()

    def get_file_status(self, file):
        if file in self.file_cards:
            return self.file_cards[file].status_label.text().replace("Status: ", "")
        return None

    def set_total_files(self, total, stages_per_file):
        self.total_stages = total * stages_per_file
        self.completed_stages = 0
        self.progress_bar.set_total_files(total, stages_per_file)

    def update_overall_progress(self, progress=None):
        if progress is None:
            if self.total_stages > 0:
                progress = min((self.completed_stages / self.total_stages) * 100, 100)
            else:
                progress = 0
        
        self.progress_bar.update_progress(progress)

    def set_delete_enabled(self, enabled):
        for card in self.file_cards.values():
            card.set_delete_enabled(enabled)

    def reset_progress(self):
        self.completed_stages = 0
        self.total_stages = 0
        self.progress_bar.reset()

    def check_existing_transcriptions(self, files):
        existing_transcriptions = []
        for file in files:
            try:
                base_name = os.path.splitext(file)[0]
                txt_path = f"{base_name}_transcription.txt"
                jsonl_path = f"{base_name}_transcription.jsonl"
                srt_path = f"{base_name}_transcription.srt"
                
                if any(os.path.exists(path) for path in [txt_path, jsonl_path, srt_path]):
                    existing_transcriptions.append(file)
            except Exception as e:
                logging.error(f"Error checking existing transcription for {file}: {str(e)}")
        
        return existing_transcriptions

class TranscribeButtonComponent(QWidget):
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_clicked.emit)
        
        self.clear_queue_stop_button = QPushButton("Stop")
        self.clear_queue_stop_button.clicked.connect(self.stop_clicked.emit)
        self.clear_queue_stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.clear_queue_stop_button)
        
        self.status_label = QLabel("")
        
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)

    def set_transcribing(self, is_transcribing):
        self.start_button.setEnabled(not is_transcribing)
        self.clear_queue_stop_button.setEnabled(is_transcribing)
        if is_transcribing:
            self.status_label.setText("Transcribing...")
        else:
            self.status_label.setText("")

    def reset(self):
        self.start_button.setEnabled(True)
        self.clear_queue_stop_button.setEnabled(False)
        self.status_label.setText("")

class ProgressBarComponent(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 20)
        layout.setSpacing(8)

        # Progress label and percentage
        label_layout = QHBoxLayout()
        self.progress_label = QLabel("Overall Progress:")
        self.progress_label.setFont(QFont("Segoe UI", 12))
        self.progress_label.setStyleSheet("color: #333;")
        
        self.percentage_label = QLabel("0%")
        self.percentage_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.percentage_label.setStyleSheet("color: #333;")
        self.percentage_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        label_layout.addWidget(self.progress_label)
        label_layout.addWidget(self.percentage_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setObjectName("overallProgressBar")

        layout.addLayout(label_layout)
        layout.addWidget(self.progress_bar)

        self.total_stages = 0
        self.completed_stages = 0

    def set_total_files(self, total, stages_per_file):
        self.total_stages = total * stages_per_file
        self.completed_stages = 0
        self.update_progress(0)

    def update_stage_progress(self, current_stage, total_stages, stage_complete):
        if current_stage > 0 and stage_complete:
            self.completed_stages += 1
            self.update_overall_progress()

    def update_overall_progress(self):
        if self.total_stages > 0:
            progress = (self.completed_stages / self.total_stages) * 100
            self.update_progress(progress)

    def update_progress(self, progress):
        progress = min(max(progress, 0), 100)  # Ensure progress is between 0 and 100
        self.progress_bar.setValue(int(progress))
        self.percentage_label.setText(f"{progress:.1f}%")

    def reset(self):
        self.total_stages = 0
        self.completed_stages = 0
        self.update_progress(0)
