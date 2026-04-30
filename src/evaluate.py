"""
evaluate.py — Full evaluation of the trained plant disease detection model.

Loads the best PyTorch checkpoint and runs inference on the test set,
reporting overall accuracy, per-class precision/recall/F1, and saving a
confusion matrix plot to docs/confusion_matrix.png.

Usage:
    conda run -n pydl python3 src/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import classification_report, confusion_matrix

from augmentation import build_val_transform

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT     = _SRC_DIR.parent
_MODELS_DIR  = _PROJECT / "models"
_DATA_DIR    = _PROJECT / "data" / "processed" / "test"
_DOCS_DIR    = _PROJECT / "docs"
_LABELS_TXT  = _MODELS_DIR / "labels.txt"
_CM_OUTPUT   = _DOCS_DIR / "confusion_matrix.png"

_CHECKPOINT_CANDIDATES = [
    _MODELS_DIR / "checkpoints" / "best_stage2.pt",
    _MODELS_DIR / "checkpoints" / "best_stage1.pt",
]

_BATCH_SIZE = 32
_CM_DPI     = 150
_CM_FIGSIZE = (8, 6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(labels_path: Path) -> list[str]:
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.txt not found at {labels_path}. Run train.py first."
        )
    return [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]


def load_checkpoint(candidates: list[Path]) -> tuple[nn.Module, Path, list[str]]:
    """Load best available checkpoint. Returns (model, path, class_names)."""
    for candidate in candidates:
        if candidate.exists():
            print(f"Loading checkpoint: {candidate}")
            ckpt = torch.load(str(candidate), map_location="cpu")
            class_names = ckpt["class_names"]
            num_classes = ckpt["num_classes"]

            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model, candidate, class_names

    paths_tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "No checkpoint found. Tried:\n  " + paths_tried + "\nRun train.py first."
    )


def build_test_loader(
    data_dir: Path,
    class_names: list[str],
) -> tuple[DataLoader, list[str]]:
    """
    Build a test DataLoader. Verifies that ImageFolder's class order matches
    the order stored in the checkpoint (which is always alphabetically sorted).
    """
    ds = datasets.ImageFolder(
        root=str(data_dir),
        transform=build_val_transform(),
    )
    actual = ds.classes
    if actual != list(class_names):
        raise RuntimeError(
            f"Class order mismatch between checkpoint and test directory.\n"
            f"  checkpoint  : {list(class_names)}\n"
            f"  test dir    : {actual}"
        )
    loader = DataLoader(ds, batch_size=_BATCH_SIZE, shuffle=False, num_workers=4)
    return loader, actual


def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_true.append(labels.numpy())
            all_pred.append(preds)

    return np.concatenate(all_true), np.concatenate(all_pred)


def short_label(class_name: str) -> str:
    if "___" in class_name:
        return class_name.split("___", maxsplit=1)[1]
    return class_name


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    short_names = [short_label(n) for n in class_names]
    n = len(short_names)

    fig, ax = plt.subplots(figsize=_CM_FIGSIZE)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    thresh = cm.max() / 2.0
    for row in range(n):
        for col in range(n):
            ax.text(
                col, row, str(cm[row, col]),
                ha="center", va="center",
                color="white" if cm[row, col] > thresh else "black",
                fontsize=8,
            )

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, pad=12)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=_CM_DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cpu")  # evaluation always on CPU for reproducibility

    model, model_path, class_names = load_checkpoint(_CHECKPOINT_CANDIDATES)
    model = model.to(device)

    test_loader, _ = build_test_loader(_DATA_DIR, class_names)

    y_true, y_pred = run_inference(model, test_loader, device)

    overall_accuracy = np.mean(y_true == y_pred) * 100.0
    missing_preds = [class_names[i] for i in range(len(class_names)) if i not in y_pred]
    if missing_preds:
        print(f"WARNING: {len(missing_preds)} class(es) with zero predictions: {missing_preds}")

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print()
    print("=== Evaluation Results ===")
    print(f"Model loaded from: {model_path}")
    print(f"Test images: {len(y_true)}")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    print()
    print("=== Per-Class Report ===")
    print(report)

    plot_confusion_matrix(cm, class_names, _CM_OUTPUT)
    print(f"=== Confusion Matrix saved to {_CM_OUTPUT} ===")


if __name__ == "__main__":
    main()
