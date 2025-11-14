import sys, time
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
import torch
import timm
from torch import nn

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QSpinBox, QCheckBox,
    QMessageBox, QFileDialog
)

# ---------- Helpers for PyInstaller onefile ----------
def resource_path(rel_path: str) -> str:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return str((base / rel_path).resolve())

DEFAULT_BUNDLED_WEIGHTS = resource_path("models/resnet18_best_from_scratch.pth")
DEFAULT_BUNDLED_CASCADE = resource_path("assets/haarcascade_frontalface_default.xml")

# ---------- Inference config ----------
IMG_SIZE = 192
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def preprocess_bgr(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)

from collections import OrderedDict
def adapt_resnet_head_keys(state: dict) -> dict:
    new_state = OrderedDict()
    for k, v in state.items():
        if k.startswith("fc.1.weight"): new_state["fc.weight"] = v
        elif k.startswith("fc.1.bias"): new_state["fc.bias"] = v
        else: new_state[k] = v
    return new_state

def load_weights_forgiving(model: nn.Module, ckpt_path: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = adapt_resnet_head_keys(ck.get("model", ck))
    head_keys = ["fc.weight","fc.bias","classifier.weight","classifier.bias","head.weight","head.bias"]
    target = dict(model.named_parameters())
    for hk in head_keys:
        if hk in state and hk in target and tuple(state[hk].shape) != tuple(target[hk].shape):
            state.pop(hk)
    model.load_state_dict(state, strict=False)

# ---------- Detectors ----------
class HaarFaceDetector:
    def __init__(self, cascade_path: str, scaleFactor=1.2, minNeighbors=5):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Cannot load cascade: {cascade_path}")
        self.scaleFactor = scaleFactor; self.minNeighbors = minNeighbors
    def detect(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors,
                                              flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
        return [(x,y,x+w,y+h) for (x,y,w,h) in faces]

# ---------- Simple Engine to reuse model & detector ----------
class EmotionEngine:
    def __init__(self, weights_path: str, cascade_path: str, use_gpu=False, fp16=False):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.fp16 = (fp16 and self.device.type == "cuda")
        self.class_names = DEFAULT_CLASSES

        self.model = timm.create_model("resnet18", pretrained=False, num_classes=len(self.class_names))
        load_weights_forgiving(self.model, weights_path)
        self.model.eval().to(self.device)
        if self.fp16: self.model.half()

        cpath = cascade_path
        if not Path(cpath).exists():
            alt = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            if Path(alt).exists(): cpath = alt
            else:
                raise FileNotFoundError("Thiếu haarcascade_frontalface_default.xml (assets/ hoặc cv2.data).")
        self.detector = HaarFaceDetector(cpath)

    def infer_bgr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """Detect largest face, classify emotion, draw overlay; return annotated frame & label text."""
        faces = self.detector.detect(frame_bgr)
        label_text = None
        if faces:
            x1,y1,x2,y2 = max(faces, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
            face = frame_bgr[y1:y2, x1:x2].copy()
            if face.size > 0:
                x = preprocess_bgr(face).to(self.device)
                if self.fp16: x = x.half()
                with torch.no_grad():
                    logits = self.model(x)
                    probs = torch.softmax(logits[0], dim=0).detach().cpu().numpy()
                k = int(np.argmax(probs)); conf = float(probs[k])
                label_text = f"{self.class_names[k]} ({conf*100:.1f}%)"
                cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame_bgr, label_text, (x1, max(20,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(frame_bgr, label_text, (x1, max(20,y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        return frame_bgr, label_text

# ---------- Video Thread ----------
class VideoThread(QThread):
    frame_ready = Signal(QImage)
    status_text = Signal(str)

    def __init__(self, engine: EmotionEngine, cam_index=0):
        super().__init__()
        self.engine = engine
        self.cam_index = cam_index
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.status_text.emit("Không mở được webcam. Hãy thử Camera Index khác.")
            return
        self.running = True
        t_prev = time.time()

        while self.running:
            ok, frame = cap.read()
            if not ok:
                self.status_text.emit("Mất khung hình từ camera.")
                break

            frame, _ = self.engine.infer_bgr(frame)

            t_now = time.time()
            fps = 1.0 / max(1e-6, t_now - t_prev); t_prev = t_now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg)

        cap.release()

    def stop(self):
        self.running = False
        self.wait(500)

# ---------- Main Window (minimal UI + Open Image) ----------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Emotion Recognition")
        self.setMinimumSize(900, 600)

        self.video_label = QLabel(alignment=Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#eee;")

        self.spn_cam = QSpinBox(); self.spn_cam.setRange(0, 10); self.spn_cam.setValue(0)
        self.chk_gpu = QCheckBox("Use GPU (CUDA)")
        self.chk_fp16 = QCheckBox("FP16")
        self.btn_open = QPushButton("📷 Mở ảnh")       
        self.btn_start = QPushButton("▶ Start")
        self.btn_stop  = QPushButton("⏹ Stop")

        # layout
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera Index:")); row.addWidget(self.spn_cam)
        row.addSpacing(12); row.addWidget(self.chk_gpu); row.addWidget(self.chk_fp16)
        row.addStretch(1); row.addWidget(self.btn_open)   # <-- thêm vào layout
        row.addWidget(self.btn_start); row.addWidget(self.btn_stop)

        root = QVBoxLayout(self)
        root.addWidget(self.video_label, 1)
        root.addLayout(row)

        self.btn_open.clicked.connect(self.on_open_image)     # <-- connect
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.thread: Optional[VideoThread] = None

        # lazy engine (không load model ngay, chỉ khi cần)
        self._engine: Optional[EmotionEngine] = None

        # cảnh báo nếu thiếu model đóng gói
        if not Path(DEFAULT_BUNDLED_WEIGHTS).exists():
            QMessageBox.warning(self, "Thiếu weights",
                                "Không tìm thấy model mặc định trong bundle. Hãy đóng gói kèm .pth vào models/.")

    # ---------- Engine ----------
    def _ensure_engine(self) -> EmotionEngine:
        if self._engine is None:
            try:
                self._engine = EmotionEngine(
                    weights_path=DEFAULT_BUNDLED_WEIGHTS,
                    cascade_path=DEFAULT_BUNDLED_CASCADE,
                    use_gpu=self.chk_gpu.isChecked(),
                    fp16=self.chk_fp16.isChecked()
                )
            except Exception as e:
                QMessageBox.critical(self, "Lỗi khởi tạo", str(e))
                raise
        return self._engine

    # ---------- Open Image (NEW) ----------
    def on_open_image(self):
        # dừng webcam nếu đang chạy để tránh “giành” khung hiển thị
        if self.thread and self.thread.isRunning():
            self.on_stop()

        file, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh",
                                              str(Path.home()),
                                              "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if not file:
            return
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, "Lỗi", "Không đọc được ảnh.")
            return

        try:
            engine = self._ensure_engine()
            annotated, label = engine.infer_bgr(img)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi suy luận", str(e))
            return

        # hiển thị ảnh kết quả
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if label:
            self.setWindowTitle(f"Face Emotion Recognition - {label}")
        else:
            self.setWindowTitle(f"Face Emotion Recognition - Không phát hiện khuôn mặt")

    # ---------- Webcam ----------
    def on_start(self):
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "Info", "Đang chạy.")
            return
        try:
            engine = self._ensure_engine()
            self.thread = VideoThread(engine=engine, cam_index=int(self.spn_cam.value()))
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e)); return
        self.thread.frame_ready.connect(self.on_frame)
        self.thread.status_text.connect(self.on_status)
        self.thread.start()

    def on_stop(self):
        if self.thread:
            self.thread.stop(); self.thread = None

    def closeEvent(self, ev):
        self.on_stop(); return super().closeEvent(ev)

    def on_frame(self, qimg: QImage):
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_status(self, msg: str):
        self.setWindowTitle(f"Face Emotion Recognition - {msg}")

# ---------- Entry ----------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
