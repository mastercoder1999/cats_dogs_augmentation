from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


@dataclass
class RunConfig:
    data_dir: str
    out_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    num_workers: int
    img_size: int
    seed: int
    amp: bool
    backbone: str
    freeze_backbone: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # ~256 pour img_size=224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return train_tf, eval_tf


def make_dataloaders(data_dir: Path, batch_size: int, num_workers: int, img_size: int) -> Tuple[Dict[str, DataLoader], Dict[str, datasets.ImageFolder]]:
    train_tf, eval_tf = get_transforms(img_size)

    ds_train = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    ds_val   = datasets.ImageFolder(data_dir / "val",   transform=eval_tf)
    ds_test  = datasets.ImageFolder(data_dir / "test",  transform=eval_tf)

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    datasets_map = {"train": ds_train, "val": ds_val, "test": ds_test}
    return loaders, datasets_map


def build_model(backbone: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    backbone = backbone.lower()

    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in model.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

        return model

    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in model.named_parameters():
                if not name.startswith("fc."):
                    p.requires_grad = False

        return model

    raise ValueError(f"Unsupported backbone: {backbone} (use resnet18 or resnet50)")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # confusion matrix (num_classes x num_classes)
    # rows = true, cols = pred
    num_classes = len(loader.dataset.classes)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += images.size(0)

        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, cm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    amp: bool
) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset/split", help="Path to split/ folder containing train/val/test")
    ap.add_argument("--out_dir", default="runs/cats_dogs_resnet", help="Output directory for model + logs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--freeze_backbone", action="store_true", help="Train only the classifier head")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    args = ap.parse_args()

    cfg = RunConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        amp=bool(args.amp),
        backbone=args.backbone,
        freeze_backbone=bool(args.freeze_backbone),
    )

    set_seed(cfg.seed)
    # FORCE NO CUDA IF BAD GPU
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < (7, 0):
            print(f"[WARN] GPU capability sm_{major}{minor} unsupported by this PyTorch build -> forcing CPU.")
            device = torch.device("cpu")

    use_amp = cfg.amp and (device.type == "cuda")


    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders, dsets = make_dataloaders(data_dir, cfg.batch_size, cfg.num_workers, cfg.img_size)

    num_classes = len(dsets["train"].classes)
    class_to_idx = dsets["train"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(cfg.backbone, num_classes=num_classes, freeze_backbone=cfg.freeze_backbone).to(device)

    # Optimizer: AdamW marche bien pour ce type de fine-tuning
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Save run config + classes mapping
    save_json(out_dir / "config.json", asdict(cfg))
    save_json(out_dir / "classes.json", {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class})

    best_val_acc = -1.0
    best_path = out_dir / "best.pt"

    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, scaler, device, use_amp)
        val_loss, val_acc, val_cm = evaluate(model, loaders["val"], device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": cfg.backbone,
                    "num_classes": num_classes,
                    "class_to_idx": class_to_idx,
                    "img_size": cfg.img_size,
                },
                best_path,
            )
            # also save confusion matrix for best val
            torch.save(val_cm, out_dir / "best_val_confusion_matrix.pt")
            print(f"  -> saved best model: {best_path} (val_acc={best_val_acc:.4f})")

        save_json(out_dir / "history.json", {"history": history})

    # ---- Final test using best model ----
    ckpt = torch.load(best_path, map_location=device)
    model = build_model(ckpt["backbone"], num_classes=ckpt["num_classes"], freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc, test_cm = evaluate(model, loaders["test"], device)
    torch.save(test_cm, out_dir / "test_confusion_matrix.pt")

    print("\n=== TEST RESULTS (best checkpoint) ===")
    print(f"test loss={test_loss:.4f} acc={test_acc:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(test_cm)

    # Save a readable metrics file
    save_json(out_dir / "final_metrics.json", {"best_val_acc": best_val_acc, "test_loss": test_loss, "test_acc": test_acc})


if __name__ == "__main__":
    main()
