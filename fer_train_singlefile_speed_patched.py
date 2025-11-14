#!/usr/bin/env python3
from __future__ import annotations
import os, math, time, glob, argparse, random, json, sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import torchvision.models as tvm


@dataclass
class CFG:
    train_dir: str = ""
    val_dir: str = ""
    ckpt_dir: str = "./checkpoints"
    best_dir: str = "./weights"
    model: str = "resnet18"
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-4
    num_workers: int = 2
    label_smoothing: float = 0.05
    resume: bool = True
    save_every_epochs: int = 1
    autosave_minutes: int = 10
    use_mixup: bool = False
    mixup_alpha: float = 0.4
    use_cutmix: bool = False
    cutmix_alpha: float = 0.8
    seed: int = 42
    log_every: int = 20



def _fmt_time(s: float) -> str:
    s = int(s)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def device_select():
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def latest_checkpoint(ckpt_dir: Path):
    files = sorted(glob.glob(str(ckpt_dir / "epoch_*.pt")))
    return files[-1] if files else None


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model, opt, sched, scaler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("opt") is not None: opt.load_state_dict(ckpt["opt"])
    if sched is not None and ckpt.get("sched") is not None: sched.load_state_dict(ckpt["sched"])
    if scaler is not None and ckpt.get("scaler") is not None: scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def build_model(name: str, num_classes: int):
    if name == "resnet18":
        m = tvm.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, num_classes))
        return m
    elif name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")


def rand_bbox(W, H, lam):
    cut_rat = (1. - lam) ** 0.5
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_data(x, y, alpha=0.4):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=0.8):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(3), x.size(2), lam)
    x2 = x[idx].clone()
    x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    y_a, y_b = y, y[idx]
    return x, y_a, y_b, lam


def mix_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


def build_transforms():
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.01, 0.04), ratio=(0.3, 3.3), value="random"),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf_train, tf_val


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_loaders_and_classes(train_dir: str, val_dir: str, bs: int, num_workers: int):
    tf_train, tf_val = build_transforms()
    ds_tr = datasets.ImageFolder(train_dir, transform=tf_train)
    ds_va = datasets.ImageFolder(val_dir, transform=tf_val)


    y = np.array([y for _, y in ds_tr.samples])
    classes = np.unique(y)
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weights = cw[y]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


    dl_kwargs = dict(
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 4

    dl_tr = DataLoader(
        ds_tr,
        sampler=sampler,
        worker_init_fn=seed_worker,
        **dl_kwargs,
    )
    dl_va = DataLoader(
        ds_va,
        shuffle=False,
        **dl_kwargs,
    )
    return dl_tr, dl_va, ds_tr.classes


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def persist_and_validate_classes(ckpt_dir: Path, class_names: list[str]):
    path = ckpt_dir / "class_names.json"
    if path.exists():
        saved = read_json(path, [])
        if saved != class_names:
            print("[ERROR] Class order mismatch!\n"
                  f"  Saved: {saved}\n"
                  f"Current: {class_names}\n"
                  "Fix: ensure your train/ and val/ folder names are IDENTICAL and sort the same.")
            sys.exit(1)
        else:
            print("[OK] Class order matches class_names.json")
    else:
        write_json(path, class_names)
        print(f"[Saved] {path}")



def train(cfg: CFG):
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)

    device = device_select()
    dl_tr, dl_va, class_names = get_loaders_and_classes(cfg.train_dir, cfg.val_dir, cfg.batch_size, cfg.num_workers)
    num_classes = len(class_names)
    print("Classes:", class_names)

    ckpt_dir = Path(cfg.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir = Path(cfg.best_dir); best_dir.mkdir(parents=True, exist_ok=True)
    persist_and_validate_classes(ckpt_dir, class_names)

    model = build_model(cfg.model, num_classes).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(dl_tr); warmup_steps = 3 * len(dl_tr)
    def lr_lambda(step):
        if step < warmup_steps: return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_path = best_dir / f"{cfg.model}_best_from_scratch.pth"

    start_epoch = 1; global_step = 0; best_acc = 0.0
    resume_batch = 0

    print("Device:", device)
    print("Steps/epoch (train):", len(dl_tr), "| Batch size:", cfg.batch_size)


    if cfg.resume:
        last = latest_checkpoint(ckpt_dir)
        if last:
            print(f"[Resume] Loading {last}")
            ck = load_checkpoint(Path(last), model, opt, sched, scaler, device)
            start_epoch = ck.get("epoch", 1)
            global_step = ck.get("global_step", 0)
            best_acc = ck.get("best_acc", 0.0)
            resume_batch = int(ck.get("batch_in_epoch", 0))
            print(f"[Resume] epoch={start_epoch} step={global_step} best_acc={best_acc:.4f} batch_in_epoch={resume_batch}")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    last_save_time = time.time()

    for ep in range(start_epoch, cfg.epochs + 1):

        torch.manual_seed(cfg.seed + ep)
        np.random.seed(cfg.seed + ep)
        random.seed(cfg.seed + ep)

        model.train(); tr_corr = tr_tot = 0
        batch_idx = 0


        data_time = step_time = batch_time = 0.0
        win = 0
        LOG_EVERY = max(1, cfg.log_every)
        cuda_sync = (device.type == "cuda")
        t_epoch_start = time.time()
        t_data_start = time.time()

        for x, y in dl_tr:

            if ep == start_epoch and resume_batch > 0 and batch_idx < resume_batch:
                batch_idx += 1
                t_data_start = time.time()
                continue


            data_dt = time.time() - t_data_start
            data_time += data_dt


            if cuda_sync: torch.cuda.synchronize()
            t_step_start = time.time()

            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            use_mix = cfg.use_mixup and random.random() < 0.5
            use_cut = (not use_mix) and cfg.use_cutmix and random.random() < 0.5
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                if use_mix:
                    x_m, y_a, y_b, lam = mixup_data(x, y, alpha=cfg.mixup_alpha)
                    logits = model(x_m); loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)
                elif use_cut:
                    x_c, y_a, y_b, lam = cutmix_data(x, y, alpha=cfg.cutmix_alpha)
                    logits = model(x_c); loss = mix_criterion(loss_fn, logits, y_a, y_b, lam)
                else:
                    logits = model(x); loss = loss_fn(logits, y)

            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

            if cuda_sync: torch.cuda.synchronize()
            step_dt = time.time() - t_step_start
            step_time += step_dt


            with torch.no_grad():
                pred = logits.argmax(1); tr_corr += (pred == y).sum().item(); tr_tot += y.size(0)

            batch_idx += 1
            global_step += 1


            batch_dt = data_dt + step_dt
            batch_time += batch_dt
            win += 1
            if (win % LOG_EVERY) == 0:
                avg_data = data_time / win
                avg_step = step_time / win
                avg_batch = batch_time / win
                imgs_per_s = (cfg.batch_size * win) / max(1e-9, batch_time)
                remaining = len(dl_tr) - batch_idx
                eta_epoch = remaining * avg_batch
                print(
                    f"[Speed][ep {ep:02d}] iter {batch_idx}/{len(dl_tr)} | "
                    f"data {avg_data:.3f}s | step {avg_step:.3f}s | batch {avg_batch:.3f}s | "
                    f"{imgs_per_s:.1f} img/s | ETA { _fmt_time(eta_epoch) } | lr={sched.get_last_lr()[0]:.2e}"
                )

                data_time = step_time = batch_time = 0.0
                win = 0


            if (time.time() - last_save_time) > cfg.autosave_minutes * 60:
                tmp = ckpt_dir / f"epoch_{ep:03d}_step_{global_step}.pt"
                save_checkpoint(tmp, {
                    "epoch": ep,
                    "global_step": global_step,
                    "best_acc": best_acc,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "batch_in_epoch": batch_idx,
                    "steps_per_epoch": len(dl_tr),
                })
                last_save_time = time.time(); print(f"[Autosave] {tmp}")


            t_data_start = time.time()

        tr_acc = tr_corr / max(1, tr_tot)
        t_epoch = time.time() - t_epoch_start
        print(f"[Epoch {ep:02d}] time={_fmt_time(t_epoch)}")


        model.eval(); va_corr = va_tot = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x); va_corr += (logits.argmax(1) == y).sum().item(); va_tot += y.size(0)
        va_acc = va_corr / max(1, va_tot)
        print(f"Epoch {ep:02d}/{cfg.epochs} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f} | lr={sched.get_last_lr()[0]:.2e}")

        if va_acc > best_acc:
            best_acc = va_acc; torch.save(model.state_dict(), best_path)
            print(f"[Best] {best_path} | best_acc={best_acc:.4f}")


        ck = ckpt_dir / f"epoch_{ep:03d}.pt"
        save_checkpoint(ck, {
            "epoch": ep + 1,
            "global_step": global_step,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "batch_in_epoch": 0,
            "steps_per_epoch": len(dl_tr),
        })
        print(f"[Checkpoint] {ck}")

        resume_batch = 0

    print(f"Done. Best val_acc={best_acc:.4f} | Best weights -> {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', type=str, required=True)
    ap.add_argument('--val_dir', type=str, required=True)
    ap.add_argument('--ckpt_dir', type=str, default=CFG.ckpt_dir)
    ap.add_argument('--best_dir', type=str, default=CFG.best_dir)
    ap.add_argument('--model', type=str, default=CFG.model)
    ap.add_argument('--epochs', type=int, default=CFG.epochs)
    ap.add_argument('--batch_size', type=int, default=CFG.batch_size)
    ap.add_argument('--lr', type=float, default=CFG.lr)
    ap.add_argument('--weight_decay', type=float, default=CFG.weight_decay)
    ap.add_argument('--num_workers', type=int, default=CFG.num_workers)
    ap.add_argument('--label_smoothing', type=float, default=CFG.label_smoothing)
    ap.add_argument('--resume', action='store_true', default=CFG.resume)
    ap.add_argument('--no-resume', dest='resume', action='store_false')
    ap.add_argument('--save_every_epochs', type=int, default=CFG.save_every_epochs)
    ap.add_argument('--autosave_minutes', type=int, default=CFG.autosave_minutes)
    ap.add_argument('--use_mixup', action='store_true', default=CFG.use_mixup)
    ap.add_argument('--use_cutmix', action='store_true', default=CFG.use_cutmix)
    ap.add_argument('--mixup_alpha', type=float, default=CFG.mixup_alpha)
    ap.add_argument('--cutmix_alpha', type=float, default=CFG.cutmix_alpha)
    ap.add_argument('--seed', type=int, default=CFG.seed)
    ap.add_argument('--log_every', type=int, default=CFG.log_every)
    args = ap.parse_args()

    cfg = CFG(**vars(args))
    train(cfg)
