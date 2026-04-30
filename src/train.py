"""
train.py — Two-stage fine-tuning of MobileNetV3Small for plant disease detection.

Stage 1: Freeze all feature extractor layers, train only the classification head
         (up to STAGE1_EPOCHS with early stopping).
Stage 2: Unfreeze the last STAGE2_UNFREEZE_PARAMS parameters of the feature
         extractor and fine-tune with a much lower learning rate.

After both stages, the best checkpoint (highest val accuracy across both stages)
is exported to ONNX format for downstream INT8 quantization (Phase 3).

Usage:
    conda run -n pydl python3 src/train.py
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models

from augmentation import build_train_transform, build_val_transform

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 32
IMAGE_SIZE: tuple[int, int] = (224, 224)

STAGE1_EPOCHS: int = 10
STAGE2_EPOCHS: int = 15

STAGE1_LR: float = 1e-3
STAGE2_LR: float = 1e-5

STAGE2_UNFREEZE_PARAMS: int = 30  # unfreeze last N named params in feature extractor

EARLY_STOP_PATIENCE: int = 5
LR_REDUCE_PATIENCE: int = 3
LR_REDUCE_FACTOR: float = 0.5
LR_MIN: float = 1e-7

RANDOM_SEED: int = 42

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
LABELS_FILE: Path = PROJECT_ROOT / "models" / "labels.txt"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "models" / "checkpoints"
BEST_CHECKPOINT_STAGE1: Path = CHECKPOINT_DIR / "best_stage1.pt"
BEST_CHECKPOINT_STAGE2: Path = CHECKPOINT_DIR / "best_stage2.pt"
ONNX_DIR: Path = PROJECT_ROOT / "models"
ONNX_FLOAT32: Path = ONNX_DIR / "plant_disease_float32.onnx"


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(labels_path: Path) -> list[str]:
    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_datasets(
    class_names: list[str],
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train and val DataLoaders using ImageFolder.

    ImageFolder sorts class directories alphabetically, which gives a
    deterministic label-to-index mapping regardless of labels.txt order.
    The sorted class list is returned so labels.txt can be updated to match.

    Returns:
        (train_loader, val_loader, sorted_class_names)
    """
    train_ds = datasets.ImageFolder(
        root=str(DATA_DIR / "train"),
        transform=build_train_transform(),
    )
    val_ds = datasets.ImageFolder(
        root=str(DATA_DIR / "val"),
        transform=build_val_transform(),
    )

    # ImageFolder.classes is always alphabetically sorted
    actual_classes = train_ds.classes

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader, actual_classes


def build_model(num_classes: int) -> tuple[nn.Module, nn.Module]:
    """
    Build MobileNetV3Small with a custom classification head.

    Returns:
        (model, features) where features is the pretrained backbone.
        features is returned so Stage 2 can selectively unfreeze parameters.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace the classifier with our custom head
    in_features = model.classifier[0].in_features  # 576 for MobileNetV3Small
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )

    # Stage 1: freeze the entire feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    return model, model.features


def save_checkpoint(
    model: nn.Module,
    val_acc: float,
    class_names: list[str],
    path: Path,
) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_accuracy": val_acc,
        "class_names": class_names,
        "num_classes": len(class_names),
    }, str(path))


def load_checkpoint(model: nn.Module, path: Path) -> float:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt["val_accuracy"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best = 0.0
        self.counter = 0

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best:
            self.best = val_acc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
) -> tuple[float, float]:
    """Run one epoch. Returns (mean_loss, accuracy)."""
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    stage: int,
    checkpoint_path: Path,
    class_names: list[str],
) -> dict:
    """Train one stage. Returns history dict with val_accuracy list."""
    early_stop = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    best_val_acc = 0.0
    history: dict[str, list] = {
        "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CHECKPOINT_DIR / f"training_stage{stage}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "lr"])

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, training=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, None, device, training=False
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, current_lr])
            f.flush()

            elapsed = time.time() - t0
            print(
                f"  Stage {stage} Epoch {epoch:3d}/{epochs}  "
                f"loss={train_loss:.4f}  acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                f"lr={current_lr:.2e}  [{elapsed:.1f}s]"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, val_acc, class_names, checkpoint_path)
                print(f"    → Saved best checkpoint (val_acc={val_acc:.4f})")

            if early_stop.step(val_acc):
                print(f"  Early stopping triggered at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    return history


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(model: nn.Module, output_path: Path, device: torch.device) -> None:
    """Export the model to ONNX format with dynamic batch axis."""
    model.eval()
    dummy_input = torch.randn(1, 3, *IMAGE_SIZE, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"ONNX model exported to: {output_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Run the full two-stage training pipeline and export the final ONNX model.

    Stage 1 — Head only (feature extractor frozen):
        Adam(lr=1e-3), up to STAGE1_EPOCHS with early stopping.

    Stage 2 — Partial unfreeze (last STAGE2_UNFREEZE_PARAMS feature extractor params):
        Adam(lr=1e-5), up to STAGE2_EPOCHS with early stopping.

    The best checkpoint (highest val accuracy across both stages) is loaded and
    exported to ONNX float32 for Phase 3 INT8 quantization.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load class names and build datasets
    # ------------------------------------------------------------------
    class_names = load_class_names(LABELS_FILE)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}) from labels.txt: {class_names}")

    train_loader, val_loader, actual_classes = build_datasets(class_names)

    # Update labels.txt to match the sorted order ImageFolder uses
    LABELS_FILE.write_text("\n".join(actual_classes) + "\n", encoding="utf-8")
    print(f"labels.txt updated to sorted order: {actual_classes}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model, features = build_model(num_classes)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 3. Stage 1 — train head only
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STAGE 1: Training classification head (feature extractor frozen)")
    print("=" * 60 + "\n")

    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR,
    )
    scheduler1 = ReduceLROnPlateau(
        optimizer1, mode="min", factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN,
    )

    history1 = train_stage(
        model, train_loader, val_loader, optimizer1, scheduler1,
        criterion, device, STAGE1_EPOCHS, stage=1,
        checkpoint_path=BEST_CHECKPOINT_STAGE1,
        class_names=actual_classes,
    )

    best_val_acc_stage1 = max(history1["val_accuracy"])
    print(
        f"\n{'=' * 60}\n"
        f"  Stage 1 complete\n"
        f"  Epochs run        : {len(history1['val_accuracy'])}\n"
        f"  Best val_accuracy : {best_val_acc_stage1:.4f} ({best_val_acc_stage1 * 100:.2f}%)\n"
        f"  Best checkpoint   : {BEST_CHECKPOINT_STAGE1}\n"
        f"{'=' * 60}\n"
    )

    # ------------------------------------------------------------------
    # 4. Stage 2 — unfreeze last N params and fine-tune
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  STAGE 2: Fine-tuning last {STAGE2_UNFREEZE_PARAMS} feature extractor params")
    print("=" * 60 + "\n")

    # Load Stage 1 best weights before unfreezing
    load_checkpoint(model, BEST_CHECKPOINT_STAGE1)
    model = model.to(device)

    # Unfreeze the last STAGE2_UNFREEZE_PARAMS named parameters of features
    all_feature_params = list(features.named_parameters())
    freeze_until = max(0, len(all_feature_params) - STAGE2_UNFREEZE_PARAMS)
    for i, (_, param) in enumerate(all_feature_params):
        param.requires_grad = (i >= freeze_until)

    unfrozen = sum(p.numel() for p in features.parameters() if p.requires_grad)
    print(f"  Unfrozen feature params: {unfrozen:,} (last {STAGE2_UNFREEZE_PARAMS} named params)")

    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE2_LR,
    )
    scheduler2 = ReduceLROnPlateau(
        optimizer2, mode="min", factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN,
    )

    history2 = train_stage(
        model, train_loader, val_loader, optimizer2, scheduler2,
        criterion, device, STAGE2_EPOCHS, stage=2,
        checkpoint_path=BEST_CHECKPOINT_STAGE2,
        class_names=actual_classes,
    )

    best_val_acc_stage2 = max(history2["val_accuracy"])
    print(
        f"\n{'=' * 60}\n"
        f"  Stage 2 complete\n"
        f"  Epochs run        : {len(history2['val_accuracy'])}\n"
        f"  Best val_accuracy : {best_val_acc_stage2:.4f} ({best_val_acc_stage2 * 100:.2f}%)\n"
        f"  Best checkpoint   : {BEST_CHECKPOINT_STAGE2}\n"
        f"{'=' * 60}\n"
    )

    # ------------------------------------------------------------------
    # 5. Load best checkpoint and export to ONNX
    # ------------------------------------------------------------------
    if best_val_acc_stage1 >= best_val_acc_stage2:
        best_checkpoint = BEST_CHECKPOINT_STAGE1
        print(
            f"Stage 1 best val_accuracy ({best_val_acc_stage1:.4f}) >= "
            f"Stage 2 ({best_val_acc_stage2:.4f}) — loading Stage 1 checkpoint."
        )
    else:
        best_checkpoint = BEST_CHECKPOINT_STAGE2
        print(
            f"Stage 2 best val_accuracy ({best_val_acc_stage2:.4f}) > "
            f"Stage 1 ({best_val_acc_stage1:.4f}) — loading Stage 2 checkpoint."
        )

    # Rebuild a fresh model to load weights cleanly
    final_model, _ = build_model(num_classes)
    # Re-enable all params so we can load Stage 2 weights (which unfroze some)
    for param in final_model.parameters():
        param.requires_grad = True
    load_checkpoint(final_model, best_checkpoint)
    final_model = final_model.to(device)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting ONNX float32 model...")
    export_to_onnx(final_model, ONNX_FLOAT32, device)

    print(
        f"\nReady for Phase 3: INT8 quantization via src/quantize.py\n"
        f"  ONNX model : {ONNX_FLOAT32}\n"
        f"  Labels     : {LABELS_FILE}\n"
    )


if __name__ == "__main__":
    train()
