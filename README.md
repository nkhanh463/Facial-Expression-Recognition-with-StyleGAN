# Emotion Desktop App (PySide6 + OpenCV + PyTorch)

This is a cross‑platform desktop GUI (Windows/macOS) for real‑time face emotion recognition using your PyTorch (timm) model.

## Project layout
```
emotion_app/
├─ main.py
├─ requirements.txt
├─ class_names.json
├─ models/
│  └─ resnet18_best_from_scratch.pth   # <- put your weights here
└─ assets/
   └─ haarcascade_frontalface_default.xml  # <- copy from your OpenCV installation
```

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python main.py
```

## Notes
- Detector defaults to **Haar Cascade** (packaging‑friendly). Copy `haarcascade_frontalface_default.xml` from your OpenCV data folder (Python: `import cv2; print(cv2.data.haarcascades)`).
- Optional MediaPipe detector: `pip install mediapipe` (you may need extra PyInstaller options later).

## Package to single executable (PyInstaller)

### Windows
```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --onefile ^
  --add-data "assets/haarcascade_frontalface_default.xml;assets" ^
  main.py
# Output: dist/main.exe
```

### macOS
```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --onefile   --add-data "assets/haarcascade_frontalface_default.xml:assets"   main.py
# Output: dist/main (or main.app)
```

If macOS Gatekeeper blocks it, right‑click → Open (first time) or codesign the app.

mac os:
pyinstaller --noconfirm --windowed --onefile \
  --add-data "assets/haarcascade_frontalface_default.xml:assets" \
  main.py


windows:
pyinstaller --noconfirm --windowed --onefile ^
  --add-data "assets\haarcascade_frontalface_default.xml;assets" ^
  main.py
