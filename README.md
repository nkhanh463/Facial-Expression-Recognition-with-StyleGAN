<<<<<<< HEAD
# Facial Expression Recognition with StyleGAN

End-to-end workflow for FER2013: generate/curate images, train a **ResNet‑18** classifier with PyTorch/timm, and deploy a PySide6 + OpenCV desktop app for real-time inference. The repo already includes trained weights, Haar Cascade, and PyInstaller config for shipping `.exe` / `.app`.
=======
# Emotion Recognition Suite (Training + Desktop)

End-to-end workflow for FER2013: generate/curate images, train a **ResNet‑18** classifier with PyTorch+timm, and deploy a PySide6 + OpenCV desktop app for real-time inference. The repo already includes trained weights, Haar Cascade, and PyInstaller config for shipping `.exe` / `.app`.
>>>>>>> 0dbe871 (Update README and add FER StyleGAN notebooks)

## Repository map

```
├─ main.py                          # PySide6 GUI: webcam + single-image inference
├─ fer_train_singlefile_speed_patched.py  # All-in-one ResNet-18 training script
├─ models/resnet18_best_from_scratch.pth  # Default desktop weights
├─ assets/haarcascade_frontalface_default.xml
├─ class_names.json                 # Fixed label order (angry → neutral)
├─ Generate.ipynb                   # StyleGAN2-ADA FER image synthesis
├─ filter_img.ipynb                 # GAN image filtering via MTCNN
├─ prj_dpl.ipynb                    # FER-2013 balancing + augmentation
├─ file_cascade.py                  # Print cv2 Haar cascade directory
└─ build/, dist/                    # PyInstaller artifacts
```

## Environment

Requirements: Python 3.10+, pip, Git, PyTorch 2.2 (CUDA 11.8 toolchain), OpenCV, PySide6. Training is GPU-oriented (CUDA or Apple MPS). The desktop app runs on CPU but benefits from CUDA + FP16 when available.

```bash
python -m venv .venv
source .venv/bin/activate          # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Tip: keep a lightweight virtualenv for the GUI and a CUDA-enabled environment (local or Colab) for heavy training/notebook work.

## Training pipeline (`fer_train_singlefile_speed_patched.py`)

- **Backbone**: ResNet‑18 only. FC head replaced with Dropout + Linear for 7 FER classes (`class_names.json`).  
- **Dataset**: ImageFolder layout (`dataset/train/<label>`, `dataset/val/<label>`) with identical label sets.  
- **Samplers & Augmentations**: WeightedRandomSampler, strong spatial/color transforms, RandomErasing.  
- **Optimization**: AdamW, cosine LR with 3‑epoch warmup, AMP, label smoothing, optional MixUp/CutMix.  
- **Resilience**: Autosave checkpoints every N minutes, full checkpoint each epoch, auto-resume from the latest `epoch_*.pt`.  
- **Export**: Best validation accuracy weights saved to `weights/resnet18_best_from_scratch.pth` → copy to `models/` for the GUI.

Example run:

```bash
python fer_train_singlefile_speed_patched.py \
  --train_dir ./dataset/train \
  --val_dir ./dataset/val \
  --epochs 40 \
  --batch_size 128 \
  --lr 1e-3 \
  --use_mixup \
  --use_cutmix \
  --autosave_minutes 10
```

Hints:
- Watch `[Speed]` logs to tune `batch_size` / `num_workers`.  
- If you change labels, delete `checkpoints/class_names.json`; the script will regenerate it.  
- Keep only the best `.pth` inside `models/` before building the desktop app.

## Desktop app (`main.py`)

Features:
- Real-time webcam stream with FPS overlay; draws the largest detected face and prints the top emotion.  
- `📷 Open Image` button for single-photo inference using the same engine.  
- Toggles for `Use GPU (CUDA)` and `FP16` must be set before starting the webcam thread.  
- Lazy model loading + QThread keep the UI responsive; `resource_path` handles PyInstaller bundles.

Run:

```bash
python main.py
```

Notes:
- Ensure `models/resnet18_best_from_scratch.pth` and `assets/haarcascade_frontalface_default.xml` exist. Use `python file_cascade.py` to locate the cascade inside OpenCV if needed.  
- `IMG_SIZE = 192` with ImageNet normalization; class order must match `class_names.json`.  
- If cascade is missing, the app tries `cv2.data.haarcascades` as a fallback.

## Packaging with PyInstaller

Fast path (inline command):

```bash
# macOS
pyinstaller --noconfirm --windowed --onefile \
  --add-data "assets/haarcascade_frontalface_default.xml:assets" \
  --add-data "models/resnet18_best_from_scratch.pth:models" \
  main.py

# Windows
pyinstaller --noconfirm --windowed --onefile ^
  --add-data "assets\\haarcascade_frontalface_default.xml;assets" ^
  --add-data "models\\resnet18_best_from_scratch.pth;models" ^
  main.py
```

`main.spec` already:
- Adds cascade + weights under `datas`.  
- Wraps the binary in a macOS bundle with `NSCameraUsageDescription`.  
- Uses `bundle_identifier='com.khanh.emotion'` (update if you need custom signing).

Artifacts live in `dist/dist/main` (single binary) or `dist/dist/main.app`. `build/` holds PyInstaller caches.

## Notebook utilities

| Notebook | Purpose |
| --- | --- |
| `Generate.ipynb` | Colab workflow: clone `stylegan2-ada-pytorch`, load a trained `.pkl`, configure `truncation_psi`, and generate ~1.3k FER images per class to Google Drive. |
| `filter_img.ipynb` | Uses `facenet-pytorch` MTCNN to keep only GAN images containing a detected face; writes to a filtered folder. |
| `prj_dpl.ipynb` | FER‑2013 balancing pipeline: analyze distribution, augment underrepresented classes with Albumentations, and write a balanced dataset (`TARGET_PER_CLASS = 3500`). |

Paths assume Colab + Google Drive; adjust `/content/drive/...` to match your workspace.

## Key assets & helpers

- `class_names.json`: canonical label ordering. Delete + retrain if your dataset changes.  
- `models/` & `assets/`: packaged by PyInstaller; keep contents minimal to shrink binaries.  
- `file_cascade.py`: prints `cv2.data.haarcascades` so you can copy the cascade into `assets/`.  
- `StyleGAN2_FER2013_Generator.ipynb`: extended StyleGAN2 workflow separate from `Generate.ipynb`.  
- `build/`, `dist/`: safe to remove before a clean rebuild.

## Suggested workflow

1. **Prepare data** – balance FER‑2013 with `prj_dpl.ipynb`, optionally augment with GAN images (`Generate.ipynb` → `filter_img.ipynb`).  
2. **Train ResNet‑18** – run `fer_train_singlefile_speed_patched.py`, monitor autosave/checkpoints, obtain `resnet18_best_from_scratch.pth`.  
3. **Verify desktop app** – copy the weight + `class_names.json`, run `python main.py`, test webcam and single-image inference.  
4. **Package** – build with PyInstaller or directly via `pyinstaller main.spec`, add codesigning if required on macOS.  
5. **Distribute** – share binaries from `dist/`, with instructions for enabling camera permissions.

## Troubleshooting

- **Missing Haar Cascade** – run `python file_cascade.py`, copy the file into `assets/`, or install `opencv-data`.  
- **Class order mismatch** – remove `checkpoints/class_names.json` if label folders changed; rerun training to regenerate.  
- **Webcam won’t open** – try another camera index; verify OS-level camera permissions (macOS: `System Settings → Privacy & Security → Camera`).  
- **CUDA unavailable** – ensure the correct PyTorch build is installed; disable the GPU checkbox to force CPU inference.  
- **FP16 instability** – only enable when CUDA is active; stick to FP32 on CPU/MPS.

## Next steps

- Experiment with additional ResNet‑18 tweaks (e.g., CutMix/MixUp schedules, optimizer tuning).  
- Integrate MediaPipe or RetinaFace detectors for better bounding boxes.  
- Add export paths (TorchScript/CoreML) if you plan to ship mobile or edge versions.  
- Write small regression scripts to benchmark ResNet‑18 on curated image sets before packaging.
