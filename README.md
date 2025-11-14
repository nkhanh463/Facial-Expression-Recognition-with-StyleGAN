📘 Facial Expression Training (PyTorch + timm)

EN: Training pipeline for Facial Expression Recognition (FER) using PyTorch + timm (ResNet-18 / MobileNetV3).
VI: Pipeline huấn luyện nhận diện cảm xúc khuôn mặt với PyTorch + timm (ResNet-18 / MobileNetV3).

📂 Project layout / Cấu trúc thư mục
fer_project/
├─ fer_train_singlefile_speed_patched.py       # main training script
├─ requirements.txt
├─ checkpoints/                                # saved epoch_xxx.pt, class_names.json
├─ weights/                                    # best .pth saved here
└─ dataset/
   ├─ train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── sad/
   │   ├── surprise/
   │   └── neutral/
   └─ val/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── sad/
       ├── surprise/
       └── neutral/


EN: Folder names under train/ and val/ must match.
VI: Tên thư mục trong train/ và val/ phải trùng nhau.

⚙️ Setup / Cài đặt môi trường
# create virtual env / tạo môi trường ảo
python -m venv venv

# activate
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# install dependencies / cài thư viện
pip install -r requirements.txt

🚀 Run training / Chạy huấn luyện
Basic command (default ResNet-18) / Lệnh cơ bản
python fer_train_singlefile_speed_patched.py \
  --train_dir ./dataset/train \
  --val_dir ./dataset/val \
  --epochs 40 \
  --batch_size 128

Full features (Mixup + CutMix + resume)
python fer_train_singlefile_speed_patched.py \
  --train_dir ./dataset/train \
  --val_dir ./dataset/val \
  --epochs 40 \
  --batch_size 128 \
  --lr 1e-3 \
  --use_mixup \
  --use_cutmix \
  --resume


TIP / Gợi ý:

Use --model mobilenet_v3_small for lightweight model.

Dùng --model mobilenet_v3_small nếu cần model nhẹ.

🔧 Important arguments / Tham số quan trọng
--train_dir          path to training dataset
--val_dir            path to validation dataset
--model              resnet18 | mobilenet_v3_small
--epochs             number of epochs (default 40)
--batch_size         batch size (default 128)
--lr                 learning rate (default 1e-3)
--lr_backbone        backbone lr (default 1e-4)
--use_mixup          enable Mixup
--use_cutmix         enable CutMix
--autosave_minutes   auto-save checkpoint every N minutes
--resume             resume training from checkpoint


EN:
Optimizer = AdamW
Scheduler = CosineAnnealingLR
Loss = CrossEntropy + label smoothing

VI:
Optimizer = AdamW
Scheduler = CosineAnnealingLR
Loss = CrossEntropy + label smoothing

📈 Outputs / Kết quả huấn luyện

Sau khi train xong:

checkpoints/
  ├─ epoch_001.pt
  ├─ epoch_002.pt
  ├─ ...
  └─ class_names.json

weights/
  └─ resnet18_best_from_scratch.pth     # best validation accuracy


EN: This .pth file is used later in your macOS / Windows / iOS apps.
VI: File .pth này dùng cho app desktop / macOS / iOS sau này.

🧪 Quick inference example / Ví dụ suy luận nhanh
import cv2, torch
import numpy as np
from torchvision import transforms
from fer_model import build_model

IMG_SIZE = 224
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

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
    k = logits.argmax(1).item()
    print("Predicted class index:", k)

📦 Applications / Ứng dụng

EN:
This trained model can be used for:

macOS desktop emotion recognition app (PySide6)

Windows PyInstaller standalone app

iOS real-time FER app (SwiftUI + CoreML)

StyleGAN augmentation

VI:
Model huấn luyện dùng được cho:

App nhận diện cảm xúc macOS (PySide6)

App Windows (PyInstaller)

App iOS real-time bằng CoreML

Tăng cường dữ liệu bằng StyleGAN
