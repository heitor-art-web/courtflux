import sys
import json
import subprocess
from pathlib import Path
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, 
    QProgressBar, QMessageBox, QPlainTextEdit, QSplitter,
    QLineEdit, QGroupBox, QFormLayout
)
from PySide6.QtGui import QFont, QPixmap

class PipelineWorker(QThread):
    """Background worker that executes analyzer.py with CLI arguments and supports clean cancellation."""
    status_update = Signal(str)
    frame_update = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, video_source: str, start_time: str, end_time: str, config_path: Path):
        super().__init__()
        self.video_source = video_source
        self.start_time = start_time
        self.end_time = end_time
        self.config_path = config_path
        self.process = None

    def run(self):
        try:
            self.status_update.emit("🚀 Initializing analysis pipeline...")
            
            analyzer_script = Path(__file__).resolve().parent / "services" / "analytics" / "analyzer.py"
            if not analyzer_script.exists():
                raise FileNotFoundError(f"Could not find analyzer script at {analyzer_script}")

            cmd = [
                sys.executable,
                str(analyzer_script),
                "--video", self.video_source,
                "--start", self.start_time,
                "--end", self.end_time
            ]

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1
            )

            for line in self.process.stdout:
                if self.isInterruptionRequested():
                    break
                line_str = line.strip()
                if not line_str:
                    continue

                if line_str.startswith("[PREVIEW]"):
                    image_path = line_str.replace("[PREVIEW]", "").strip()
                    self.frame_update.emit(image_path)
                else:
                    self.status_update.emit(line_str)

            if self.process:
                self.process.wait()

            if self.isInterruptionRequested():
                return

            if self.process.returncode != 0:
                raise RuntimeError(f"analyzer.py exited with code {self.process.returncode}")

            self.finished.emit("✅ Analysis pipeline finished successfully!")

        except Exception as e:
            if not self.isInterruptionRequested():
                self.error.emit(str(e))

    def cancel(self):
        """Gracefully terminates the thread and the underlying python process."""
        self.requestInterruption()
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Basketball Analytics Suite")
        self.resize(1000, 680)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 1. Match Info Header
        self.info_label = QLabel("Loading match configuration...")
        main_layout.addWidget(self.info_label)
        
        # 2. Video Input & Time Trim Controls
        input_group = QGroupBox("Video Input Settings (All fields required)")
        input_layout = QVBoxLayout(input_group)

        # Video Source Picker / URL Input
        source_layout = QHBoxLayout()
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Paste YouTube URL or select a local video file...")
        self.source_input.textChanged.connect(self.validate_inputs)
        
        browse_btn = QPushButton("Browse Local Video...")
        browse_btn.clicked.connect(self.select_video)
        
        source_layout.addWidget(self.source_input, stretch=3)
        source_layout.addWidget(browse_btn, stretch=1)
        input_layout.addLayout(source_layout)

        # Start and End Time Trimming
        time_form = QFormLayout()
        
        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("e.g., 0 or 00:00")
        self.start_time_input.textChanged.connect(self.validate_inputs)
        
        self.end_time_input = QLineEdit()
        self.end_time_input.setPlaceholderText("e.g., 120 or 02:00")
        self.end_time_input.textChanged.connect(self.validate_inputs)

        time_form.addRow("Start Time (sec / MM:SS): *", self.start_time_input)
        time_form.addRow("End Time (sec / MM:SS): *", self.end_time_input)
        
        input_layout.addLayout(time_form)
        main_layout.addWidget(input_group)
        
        # 3. Execution Control
        self.run_btn = QPushButton("🚀 Run Analysis Pipeline")
        self.run_btn.setEnabled(False) # Disabled by default until validation passes
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.start_pipeline)
        main_layout.addWidget(self.run_btn)

        # 4. Progress Indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 5. Splitter: Live Preview (Left) + Console Logs (Right)
        content_splitter = QSplitter(Qt.Horizontal)

        # Left: Frame Preview
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(QLabel("<b>Live Video Analysis Preview:</b>"))

        self.preview_label = QLabel("Video preview will appear here during analysis...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #000; color: #888; border-radius: 4px; min-height: 280px;")
        preview_layout.addWidget(self.preview_label)
        content_splitter.addWidget(preview_container)

        # Right: Terminal Console
        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("<b>Live Execution Logs:</b>"))

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setFont(QFont("Consolas", 10))
        self.log_console.setStyleSheet("""
            QPlainTextEdit {
                background-color: #1e1e1e;
                color: #00ff66;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px;
            }
        """)
        log_layout.addWidget(self.log_console)
        content_splitter.addWidget(log_container)
        
        content_splitter.setSizes([500, 500])
        main_layout.addWidget(content_splitter)
        
        self.load_match_config()

    def load_match_config(self):
        try:
            config_path = Path(__file__).resolve().parent / "data" / "match_config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                metadata = config.get("metadata", {})
                home, away = metadata.get("home_team"), metadata.get("away_team")

                if home and away:
                    self.info_label.setText(f"🏟️ Loaded Match Config: <b>{home}</b> vs <b>{away}</b>")
                else:
                    teams = list(config.get("teams", {}).keys())
                    self.info_label.setText(f"🏟️ Loaded Teams: {', '.join(teams)}")
            else:
                self.info_label.setText("⚠️ Warning: match_config.json not found in data/ directory.")
        except Exception as e:
            self.info_label.setText(f"❌ Error loading configuration: {e}")

    def select_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Basketball Game Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_name:
            self.source_input.setText(file_name)

    def validate_inputs(self):
        """Validates that Source, Start Time, and End Time are all populated."""
        has_source = len(self.source_input.text().strip()) > 0
        has_start = len(self.start_time_input.text().strip()) > 0
        has_end = len(self.end_time_input.text().strip()) > 0
        
        # Only enable if NOT currently running a worker and all fields have text
        is_running = hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning()
        if not is_running:
            self.run_btn.setEnabled(has_source and has_start and has_end)

    def start_pipeline(self):
        # If already running, treat button click as an abort request
        if hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning():
            self.append_log("\n🛑 Aborting pipeline...")
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Stopping...")
            self.worker.cancel()
            self.worker.wait()
            self.finish_pipeline("Pipeline aborted by user.")
            return

        # Start sequence: configure button as "Cancel" toggle
        self.run_btn.setText("🛑 Cancel Analysis")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #d9534f; color: white;")
        self.run_btn.setEnabled(True) # Keep enabled so user can click it to cancel
        self.progress_bar.setVisible(True)
        self.log_console.clear()
        
        video_source = self.source_input.text().strip()
        start_time = self.start_time_input.text().strip()
        end_time = self.end_time_input.text().strip()
        config_path = Path(__file__).resolve().parent / "data" / "match_config.json"
        
        self.worker = PipelineWorker(video_source, start_time, end_time, config_path)
        self.worker.status_update.connect(self.append_log)
        self.worker.frame_update.connect(self.update_preview)
        self.worker.finished.connect(self.finish_pipeline)
        self.worker.error.connect(self.on_pipeline_error)
        self.worker.start()

    @Slot(str)
    def append_log(self, message: str):
        self.log_console.appendPlainText(message)
        self.log_console.ensureCursorVisible()

    @Slot(str)
    def update_preview(self, image_path: str):
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)

    @Slot(str)
    def finish_pipeline(self, message: str = "Analysis complete!"):
        self.progress_bar.setVisible(False)
        self.run_btn.setText("🚀 Run Analysis Pipeline")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        
        # Re-enable the run button based on current input text validation
        self.validate_inputs()
        
        self.append_log(f"\n{message}")
        
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if "aborted" in message.lower():
            return

        QMessageBox.information(self, "Success", "Analysis finished! Launching dashboard...")
        
        try:
            dashboard_path = Path(__file__).resolve().parent / "ui" / "dashboard.py"
            if dashboard_path.exists():
                kwargs = {}
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
                    
                subprocess.Popen(
                    [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
                    **kwargs
                )
            else:
                QMessageBox.warning(self, "Missing Dashboard", f"Could not find dashboard file at:\n{dashboard_path}")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to open dashboard:\n{e}")

    @Slot(str)
    def on_pipeline_error(self, err_msg: str):
        self.progress_bar.setVisible(False)
        self.run_btn.setText("🚀 Run Analysis Pipeline")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.validate_inputs()
        self.append_log(f"\n❌ Pipeline Failure:\n{err_msg}")
        QMessageBox.critical(self, "Execution Error", f"An error occurred during processing:\n\n{err_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())