# gui/dialogs.py
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QCheckBox, QPushButton

class FolderSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Folder Selection")
        self.setFixedSize(300, 150)  # Set a fixed size for the dialog
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Add a label
        label = QLabel("Select folder options:")
        label.setObjectName("dialogLabel")
        layout.addWidget(label)
        
        self.include_subfolders_check = QCheckBox("Include Subfolders")
        self.include_subfolders_check.setObjectName("dialogCheckBox")
        layout.addWidget(self.include_subfolders_check)
        
        # Add some spacing
        layout.addStretch()
        
        # Button layout
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.ok_button.setObjectName("dialogButton")
        self.cancel_button.setObjectName("dialogButton")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def include_subfolders(self):
        return self.include_subfolders_check.isChecked()

class IntegratedFileDialog(QDialog):
    def __init__(self, existing_transcriptions_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Existing Transcriptions")
        self.setFixedWidth(400)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        info_label = QLabel(f"{existing_transcriptions_count} file(s) already have transcriptions.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #333333; font-size: 14px; font-weight: bold;")
        layout.addWidget(info_label)
        
        question_label = QLabel("Would you like to redo these transcriptions?")
        question_label.setWordWrap(True)
        question_label.setStyleSheet("color: #333333; font-size: 14px;")
        layout.addWidget(question_label)
        
        button_layout = QHBoxLayout()
        self.yes_button = QPushButton("Yes")
        self.no_button = QPushButton("No")
        button_layout.addWidget(self.yes_button)
        button_layout.addWidget(self.no_button)
        layout.addLayout(button_layout)
        
        self.yes_button.clicked.connect(self.accept)
        self.no_button.clicked.connect(self.reject)

    def exec(self):
        return super().exec()