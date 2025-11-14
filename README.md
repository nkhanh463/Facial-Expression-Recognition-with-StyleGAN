# Facial Expression Recognition — Training Pipeline

Complete PyTorch + timm training stack for Facial Expression Recognition (FER) with ResNet‑18 or MobileNetV3‑Small backbones. Includes class-balanced sampling, aggressive augmentations, MixUp/CutMix, cosine LR with warmup, AMP, autosave, and ready-to-export `.pth` weights for desktop or mobile apps.

## Features
- End-to-end training script with resume + autosave checkpoints
- ImageFolder data loading with face-safe augmentations and balanced sampler
- MixUp / CutMix with label smoothing for stable optimization
- Cosine decay scheduler, AdamW optimizer, AMP mixed precision
- Best-weight export for PySide6 desktop, PyInstaller, or CoreML apps

## Project Layout
```text
fer_project/
├─ fer_train_singlefile_speed_patched.py
├─ requirements.txt
├─ checkpoints/             # auto-generated (epoch_xxx.pt, class_names.json)
├─ weights/                 # best .pth saved here
└─ dataset/
   ├─ train/
   │   └─ emotion folders...
   └─ val/
       └─ emotion folders...
```

## Dataset Structure
Emotion folders must be identical between `train/` and `val/`. Supported labels:

```
angry, disgust, fear, happy, sad, surprise, neutral
```

## Setup
```bash
python -m venv venv
```

Activate the environment:

```powershell
# Windows
venv\Scripts\activate
```

```bash
# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training
### Basic (ResNet-18 defaults)
```bash
python fer_train_singlefile_speed_patched.py \
  --train_dir ./dataset/train \
  --val_dir ./dataset/val \
  --epochs 40 \
  --batch_size 128
```

### MixUp + CutMix + Resume
```bash
python fer_train_singlefile_speed_patched.py \
  --train_dir ./dataset/train \
  --val_dir ./dataset/val \
  --epochs 40 \
  --batch_size 128 \
  --lr 1e-3 \
  --use_mixup \
  --use_cutmix \
  --resume
```

### Use MobileNetV3-Small
```bash
python fer_train_singlefile_speed_patched.py ... --model mobilenet_v3_small
```

## Key Arguments

| Argument | Description |
| --- | --- |
| `--train_dir` | Path to training ImageFolder |
| `--val_dir` | Path to validation ImageFolder |
| `--model` | `resnet18` (default) or `mobilenet_v3_small` |
| `--epochs` | Number of training epochs (default 40) |
| `--batch_size` | Batch size (default 128) |
| `--lr` | Learning rate (default `1e-3`) |
| `--use_mixup` / `--use_cutmix` | Enable MixUp or CutMix |
| `--autosave_minutes` | Autosave interval in minutes |
| `--resume` | Resume from the most recent checkpoint |

## Output Files
After training, you will have:

```text
checkpoints/
  ├─ epoch_001.pt
  ├─ epoch_002.pt
  └─ class_names.json

weights/
  └─ resnet18_best_from_scratch.pth
```

`resnet18_best_from_scratch.pth` can be consumed by PySide6 desktop apps, PyInstaller executables, or exported to CoreML for iOS (SwiftUI) apps.

## Quick Inference Example
```python
import cv2, torch
import numpy as np
from torchvision import transforms
from fer_model import build_model

IMG_SIZE = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

img = cv2.imread("face.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = tf(img).unsqueeze(0)

model = build_model("resnet18", 7)
state = torch.load("weights/resnet18_best_from_scratch.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

with torch.no_grad():
    logits = model(img)
    cls = logits.argmax(1).item()
    print("Predicted class:", cls)
```

## Notes
- Optimizer: **AdamW**
- Scheduler: **CosineAnnealingLR** (warmup implemented via LambdaLR)
- Loss: **CrossEntropy** with label smoothing
- AMP mixed precision for faster training on CUDA
- Autosave prevents progress loss during long runs
