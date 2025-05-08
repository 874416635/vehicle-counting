import subprocess
import cv2
from process_picture_with_YOLO import count_vehicles_picture
from process_video_with_YOLO import count_vehicles_video
import os
import sqlite3
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QMessageBox, QFileDialog, QHBoxLayout, QListWidget, QMainWindow,
    QDialog, QProgressBar, QStackedWidget, QListWidgetItem, QSplitter
)
import sys


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Vehicle Counting System")
        self.setGeometry(100, 100, 400, 300)
        self.init_ui()
        self.setStyleSheet("""
            QWidget {
                font-family: 'Microsoft YaHei';
            }
        """)

    def init_ui(self):
        # Create stacked layout
        self.stacked_widget = QStackedWidget()
        self.login_widget = self.create_login_form()
        self.register_widget = self.create_register_form()

        self.stacked_widget.addWidget(self.login_widget)
        self.stacked_widget.addWidget(self.register_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.setLayout(main_layout)

        # Initially show login page
        self.show_login()

    def create_login_form(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Username input
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        layout.addWidget(self.username_input)

        # Password input
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Password")
        layout.addWidget(self.password_input)

        # Login button
        login_btn = QPushButton("Login")
        login_btn.setCursor(Qt.PointingHandCursor)
        login_btn.clicked.connect(self.check_credentials)
        layout.addWidget(login_btn)

        # Register link
        register_link = QPushButton("No account? Register now")
        register_link.setStyleSheet("text-align: left; border: none; color: blue;")
        register_link.setCursor(Qt.PointingHandCursor)
        register_link.clicked.connect(self.show_register)
        layout.addWidget(register_link)

        widget.setLayout(layout)
        return widget

    def create_register_form(self):
        widget = QWidget()
        layout = QVBoxLayout()

        # Username input
        self.reg_username = QLineEdit()
        self.reg_username.setPlaceholderText("Username")
        layout.addWidget(self.reg_username)

        # Password input
        self.reg_password = QLineEdit()
        self.reg_password.setEchoMode(QLineEdit.Password)
        self.reg_password.setPlaceholderText("Password")
        layout.addWidget(self.reg_password)

        # Confirm password
        self.confirm_password = QLineEdit()
        self.confirm_password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setPlaceholderText("Confirm Password")
        layout.addWidget(self.confirm_password)

        # Register button
        register_btn = QPushButton("Register")
        register_btn.clicked.connect(self.register_user)
        layout.addWidget(register_btn)

        # Back to login link
        back_link = QPushButton("Already have an account? Back to login")
        back_link.setStyleSheet("text-align: left; border: none; color: blue;")
        back_link.setCursor(Qt.PointingHandCursor)
        back_link.clicked.connect(self.show_login)
        layout.addWidget(back_link)

        widget.setLayout(layout)
        return widget

    def show_login(self):
        self.setFixedSize(300, 200)
        self.stacked_widget.setCurrentIndex(0)

    def show_register(self):
        self.setFixedSize(300, 280)
        self.stacked_widget.setCurrentIndex(1)

    def check_credentials(self):
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password cannot be empty!")
            return

        conn = sqlite3.connect('user.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?",
                       (username, password))
        result = cursor.fetchone()
        conn.close()

        if result:
            self.main_window = MainWindow(username)
            self.main_window.show()
            self.close()
        else:
            QMessageBox.warning(self, "Error", "Invalid username or password!")

    def register_user(self):
        username = self.reg_username.text()
        password = self.reg_password.text()
        confirm_pwd = self.confirm_password.text()

        if not all([username, password, confirm_pwd]):
            QMessageBox.warning(self, "Error", "Please fill in all fields!")
            return

        if password != confirm_pwd:
            QMessageBox.warning(self, "Error", "Passwords do not match!")
            return

        # Check if username exists
        conn = sqlite3.connect('user.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        if cursor.fetchone():
            QMessageBox.warning(self, "Error", "Username already exists!")
            conn.close()
            return

        # Create user directories
        user_dir = os.path.join("user_data", username)
        try:
            os.makedirs(os.path.join(user_dir, "pictures"), exist_ok=True)
            os.makedirs(os.path.join(user_dir, "videos"), exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create user directories: {str(e)}")
            return

        # Write to database
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                           (username, password))
            conn.commit()
            conn.close()
            QMessageBox.information(self, "Success", "Registration successful!")
            self.show_login()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Registration failed: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self, username):
        super().__init__()
        self.username = username
        self.setWindowTitle("Smart Vehicle Counting System")
        self.setGeometry(100, 100, 800, 600)
        self.current_page = "Images"  # Default page
        self.init_ui()

    def init_ui(self):
        # Left navigation bar
        self.nav_bar = QListWidget()
        self.nav_bar.addItems(["Images", "Videos", "History", "Logout"])
        self.nav_bar.setMaximumWidth(150)
        self.nav_bar.currentRowChanged.connect(self.change_page)

        # Right content area
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout()

        # Top area: display processed images
        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setStyleSheet("border: 1px solid black;")
        self.right_layout.addWidget(self.image_view, 3)

        # Bottom area: button layout
        button_layout = QHBoxLayout()

        # Upload button
        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_file)
        button_layout.addWidget(self.upload_button, 1)

        # Batch upload button
        self.batch_upload_button = QPushButton("Batch Upload")
        self.batch_upload_button.clicked.connect(self.batch_upload_files)
        button_layout.addWidget(self.batch_upload_button, 1)

        self.right_layout.addLayout(button_layout, 1)

        # Set right content layout
        self.right_widget.setLayout(self.right_layout)

        # Main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.nav_bar, 1)
        main_layout.addWidget(self.right_widget, 3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def change_page(self, index):
        # Handle logout
        if index == 3:
            self.logout()
            return

        # Clear old content
        while self.right_layout.count():
            child = self.right_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Rebuild content based on page
        pages = ["Images", "Videos", "History"]
        self.current_page = pages[index]

        if self.current_page in ["Images", "Videos"]:
            self.rebuild_upload_interface()
        else:
            self.build_history_interface()

    def rebuild_upload_interface(self):
        """Rebuild image/video upload interface"""
        self.image_view = QLabel()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setStyleSheet("border: 1px solid black;")
        self.right_layout.addWidget(self.image_view, 3)

        button_layout = QHBoxLayout()
        self.upload_button = QPushButton(f"Upload {self.current_page}")
        self.upload_button.clicked.connect(self.upload_file)
        button_layout.addWidget(self.upload_button)

        self.batch_upload_button = QPushButton("Batch Upload")
        self.batch_upload_button.clicked.connect(self.batch_upload_files)
        button_layout.addWidget(self.batch_upload_button)

        self.right_layout.addLayout(button_layout, 1)

    def build_history_interface(self):
        """Build history interface"""
        splitter = QSplitter(Qt.Horizontal)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.show_history_preview)
        splitter.addWidget(self.history_list)

        self.load_history_data()

        self.right_layout.addWidget(splitter)

    def load_history_data(self):
        """Load user history data"""
        self.history_list.clear()

        # Get image history
        pic_dir = os.path.join("user_data", self.username, "pictures")
        for f in os.listdir(pic_dir):
            if "_result" in f:
                item = QListWidgetItem(f"📷 {f}")
                item.file_path = os.path.join(pic_dir, f)
                self.history_list.addItem(item)

        # Get video history
        video_dir = os.path.join("user_data", self.username, "videos")
        for f in os.listdir(video_dir):
            if "_result" in f:
                item = QListWidgetItem(f"🎥 {f}")
                item.file_path = os.path.join(video_dir, f)
                self.history_list.addItem(item)

    def show_history_preview(self, item):
        """Open file when clicking history item"""
        file_path = item.file_path
        if os.path.exists(file_path):
            try:
                if sys.platform.startswith('darwin'):  # macOS
                    subprocess.run(['open', file_path], check=True)
                elif os.name == 'nt':  # Windows
                    os.startfile(file_path)
                elif os.name == 'posix':  # Linux
                    subprocess.run(['xdg-open', file_path], check=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot open file: {str(e)}")
        else:
            QMessageBox.warning(self, "Warning", "File does not exist!")

    def logout(self):
        self.close()
        self.login_window = LoginWindow()
        self.login_window.show()

    def upload_file(self):
        if self.current_page == "Images":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
            if file_path:
                self.process_single_image(file_path)
        elif self.current_page == "Videos":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi)")
            if file_path:
                self.process_single_video(file_path)

    def batch_upload_files(self):
        if self.current_page == "Images":
            file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
            if file_paths:
                self.process_batch_images(file_paths)
        elif self.current_page == "Videos":
            file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Videos", "", "Videos (*.mp4 *.avi)")
            if file_paths:
                self.process_batch_videos(file_paths)

    def process_single_image(self, file_path):
        try:
            user_dir = os.path.join("user_data", self.username, "pictures")
            os.makedirs(user_dir, exist_ok=True)
            base_name, ext = os.path.splitext(os.path.basename(file_path))
            output_path = os.path.join(user_dir, f"{base_name}_result{ext}")

            count_vehicles_picture(file_path, output_path)
            self.show_processed_image(output_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image: {e}")

    def process_batch_images(self, file_paths):
        try:
            output_dir = os.path.join("user_data", self.username, "pictures")
            os.makedirs(output_dir, exist_ok=True)

            self.progress_dialog = ProgressDialog(len(file_paths), self)
            self.progress_dialog.cancel_button.clicked.connect(self.cancel_batch_processing)

            self.batch_worker = BatchWorker(file_paths, output_dir)
            self.batch_worker.progress_update.connect(self.update_progress)
            self.batch_worker.finished_signal.connect(self.images_batch_processing_finished)
            self.batch_worker.start()

            self.progress_dialog.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch processing initialization failed: {e}")

    def process_single_video(self, file_path):
        try:
            user_dir = os.path.join("user_data", self.username, "videos")
            os.makedirs(user_dir, exist_ok=True)
            base_name, ext = os.path.splitext(os.path.basename(file_path))
            output_path = os.path.join(user_dir, f"{base_name}_result{ext}")

            self.progress_dialog = ProgressDialog(0, self)
            self.progress_dialog.show()

            self.single_video_worker = SingleVideoWorker(file_path, output_path)
            self.single_video_worker.finished_signal.connect(
                lambda out: self.single_video_finished(out))
            self.single_video_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing video: {e}")

    def process_batch_videos(self, file_paths):
        try:
            output_dir = os.path.join("user_data", self.username, "videos")
            os.makedirs(output_dir, exist_ok=True)

            self.progress_dialog = ProgressDialog(len(file_paths), self)
            self.progress_dialog.cancel_button.clicked.connect(self.cancel_batch_processing)

            self.video_batch_worker = VideoBatchWorker(file_paths, output_dir)
            self.video_batch_worker.progress_update.connect(self.update_progress)
            self.video_batch_worker.finished_signal.connect(self.videos_batch_processing_finished)
            self.video_batch_worker.start()

            self.progress_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Batch processing initialization failed: {e}")

    def update_progress(self, current, filename):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.update_progress(current, filename)

    def single_video_finished(self, output_path):
        if output_path:
            cap = cv2.VideoCapture(output_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, last_frame = cap.read()
            cap.release()
            if ret:
                temp_img_path = output_path.replace(".mp4", "_last.jpg")
                cv2.imwrite(temp_img_path, last_frame)
                pixmap = QPixmap(temp_img_path)
                self.image_view.setPixmap(pixmap.scaled(
                    self.image_view.width(),
                    self.image_view.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))
                try:
                    os.remove(temp_img_path)
                except Exception as e:
                    print(f"Failed to delete temp file: {e}")
            QMessageBox.information(self, "Complete", "Video processing completed")
        else:
            QMessageBox.critical(self, "Error", "Video processing failed")
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

    def images_batch_processing_finished(self, processed_files):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.accept()
        if processed_files:
            QMessageBox.information(self, "Complete", f"Successfully processed {len(processed_files)} images")
            self.show_processed_image(processed_files[-1])

    def videos_batch_processing_finished(self, processed_files):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.accept()

        if processed_files:
            last_video_path = processed_files[-1]
            frame_paths = []
            try:
                cap = cv2.VideoCapture(last_video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                    ret, last_frame = cap.read()
                    if ret:
                        temp_img_path = f"{os.path.splitext(last_video_path)[0]}_last.jpg"
                        cv2.imwrite(temp_img_path, last_frame)
                        frame_paths.append(temp_img_path)
                cap.release()
            except Exception as e:
                print(f"Frame extraction failed: {last_video_path} - {str(e)}")

            if frame_paths:
                self.show_processed_image(frame_paths[0])
                try:
                    os.remove(frame_paths[0])
                except Exception as e:
                    print(f"Failed to delete temp file: {e}")

            QMessageBox.information(self, "Complete", f"Successfully processed {len(processed_files)} videos")
        else:
            QMessageBox.critical(self, "Error", "Video processing failed")

    def cancel_batch_processing(self):
        if hasattr(self, 'batch_worker'):
            self.batch_worker.stop()
            QMessageBox.warning(self, "Cancelled", "Processing cancelled")

    def show_processed_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_view.setPixmap(pixmap.scaled(
            self.image_view.width(),
            self.image_view.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))


class ProgressDialog(QDialog):
    def __init__(self, total_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Progress")
        self.setFixedSize(400, 150)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        layout = QVBoxLayout()

        self.progress_bar = QProgressBar(self)
        if total_files == 0:
            self.progress_bar.setRange(0, 0)
            self.status_label = QLabel("Processing video...", self)
        else:
            self.progress_bar.setMaximum(total_files)
            self.status_label = QLabel("Processing file 0/{}".format(total_files), self)

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        self.setLayout(layout)

    def update_progress(self, current, filename):
        if self.progress_bar.maximum() == 0:
            return
        self.progress_bar.setValue(current-1)
        self.status_label.setText(f"Processing: {filename} ({current}/{self.progress_bar.maximum()})")


class BatchWorker(QThread):
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(list)

    def __init__(self, file_paths, output_dir):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self._is_running = True

    def run(self):
        processed_files = []
        for i, file_path in enumerate(self.file_paths):
            if not self._is_running:
                break
            try:
                base_name, ext = os.path.splitext(os.path.basename(file_path))
                output_path = os.path.join(self.output_dir, f"{base_name}_result{ext}")
                count_vehicles_picture(file_path, output_path)
                processed_files.append(output_path)
                self.progress_update.emit(i + 1, os.path.basename(file_path))
            except Exception as e:
                print(f"Processing failed: {file_path} - {str(e)}")
        self.finished_signal.emit(processed_files)

    def stop(self):
        self._is_running = False


class SingleVideoWorker(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, file_path, output_path):
        super().__init__()
        self.file_path = file_path
        self.output_path = output_path

    def run(self):
        try:
            count_vehicles_video(self.file_path, self.output_path)
            self.finished_signal.emit(self.output_path)
        except Exception as e:
            print(f"Processing failed: {str(e)}")
            self.finished_signal.emit(None)


class VideoBatchWorker(QThread):
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(list)

    def __init__(self, file_paths, output_dir):
        super().__init__()
        self.file_paths = file_paths
        self.output_dir = output_dir
        self._is_running = True

    def run(self):
        processed_files = []
        for i, file_path in enumerate(self.file_paths):
            if not self._is_running:
                break
            try:
                base_name, ext = os.path.splitext(os.path.basename(file_path))
                output_path = os.path.join(self.output_dir, f"{base_name}_result{ext}")

                self.progress_update.emit(i + 1, os.path.basename(file_path))

                count_vehicles_video(file_path, output_path)
                processed_files.append(output_path)

            except Exception as e:
                print(f"Processing failed: {file_path} - {str(e)}")
        self.finished_signal.emit(processed_files)

    def stop(self):
        self._is_running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())