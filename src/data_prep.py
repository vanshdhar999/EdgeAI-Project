"""
data_prep.py — PlantVillage dataset preparation for Edge AI plant disease detection.

Selects a balanced 5-class tomato subset, resizes images to 224x224,
performs a stratified 70/15/15 train/val/test split, and generates labels.txt.

Usage:
    python src/data_prep.py

Outputs:
    data/processed/train/<class>/   — training images
    data/processed/val/<class>/     — validation images
    data/processed/test/<class>/    — test images
    models/labels.txt               — class index to disease name mapping
"""

import shutil
import random
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root of the raw PlantVillage color images
RAW_DATASET_DIR = Path("plantvillage-dataset/color")

# Output directories
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# Target image size for training
IMAGE_SIZE = (224, 224)

# Train / val / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Classes selected for the initial pipeline validation.
# Chosen from the tomato plant family for visual consistency; sizes range
# 1,000–2,127 images each to keep class balance reasonable.
SELECTED_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(class_dir: Path) -> list[Path]:
    """Return a sorted list of valid image paths inside class_dir."""
    return sorted(
        p for p in class_dir.iterdir()
        if p.suffix in IMAGE_EXTENSIONS
    )


def stratified_split(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split a list of paths into train / val / test subsets.

    Splits are deterministic given the same seed and preserve the relative
    ordering within each split (shuffle happens before splitting).

    Returns:
        (train_paths, val_paths, test_paths)
    """
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def resize_and_copy(src: Path, dst: Path) -> None:
    """
    Open src image, resize to IMAGE_SIZE using Lanczos resampling,
    convert to RGB (drops alpha / grayscale), and save to dst as JPEG.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.LANCZOS)
        # Always save as .jpg regardless of source extension
        img.save(dst.with_suffix(".jpg"), format="JPEG", quality=95)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_dataset() -> None:
    """
    Full Phase 1 data preparation pipeline:
      1. Validate that all selected class directories exist in the raw dataset.
      2. Collect image paths per class.
      3. Perform a stratified 70/15/15 split.
      4. Resize and copy images into data/processed/{train,val,test}/<class>/.
      5. Write models/labels.txt (index → class name).
      6. Print a summary report.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # Step 1 — Validate class directories
    # ------------------------------------------------------------------
    print("=== Phase 1: Data Preparation ===\n")
    print(f"Raw dataset: {RAW_DATASET_DIR.resolve()}")
    print(f"Image size:  {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"Split:       {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)} (train/val/test)")
    print(f"Classes:     {len(SELECTED_CLASSES)}\n")

    for class_name in SELECTED_CLASSES:
        class_dir = RAW_DATASET_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found: {class_dir}\n"
                f"Available classes: {[d.name for d in RAW_DATASET_DIR.iterdir() if d.is_dir()]}"
            )

    # ------------------------------------------------------------------
    # Step 2 — Collect images and run split
    # ------------------------------------------------------------------
    split_registry: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    stats: dict[str, dict[str, int]] = {}

    for class_name in SELECTED_CLASSES:
        class_dir = RAW_DATASET_DIR / class_name
        all_images = collect_images(class_dir)

        if len(all_images) == 0:
            raise ValueError(f"No images found in {class_dir}")

        train, val, test = stratified_split(
            all_images, TRAIN_RATIO, VAL_RATIO, seed=RANDOM_SEED
        )

        split_registry["train"][class_name] = train
        split_registry["val"][class_name] = val
        split_registry["test"][class_name] = test

        stats[class_name] = {
            "total": len(all_images),
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }

    # ------------------------------------------------------------------
    # Step 3 — Verify no data leakage across splits
    # ------------------------------------------------------------------
    print("Verifying no data leakage across splits...")
    for class_name in SELECTED_CLASSES:
        train_names = {p.stem for p in split_registry["train"][class_name]}
        val_names   = {p.stem for p in split_registry["val"][class_name]}
        test_names  = {p.stem for p in split_registry["test"][class_name]}

        tv_overlap = train_names & val_names
        tt_overlap = train_names & test_names
        vt_overlap = val_names & test_names

        if tv_overlap or tt_overlap or vt_overlap:
            raise AssertionError(
                f"Data leakage detected in class '{class_name}':\n"
                f"  train∩val={tv_overlap}\n"
                f"  train∩test={tt_overlap}\n"
                f"  val∩test={vt_overlap}"
            )
    print("  [OK] No leakage detected.\n")

    # ------------------------------------------------------------------
    # Step 4 — Resize and copy images
    # ------------------------------------------------------------------
    total_images = sum(s["total"] for s in stats.values())
    processed = 0

    for split_name in ("train", "val", "test"):
        for class_name, paths in split_registry[split_name].items():
            for src in paths:
                dst = PROCESSED_DIR / split_name / class_name / src.name
                resize_and_copy(src, dst)
                processed += 1
                if processed % 500 == 0:
                    print(f"  Processed {processed}/{total_images} images...")

    print(f"  Processed {processed}/{total_images} images — done.\n")

    # ------------------------------------------------------------------
    # Step 5 — Write labels.txt
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = MODELS_DIR / "labels.txt"
    with labels_path.open("w") as f:
        for idx, class_name in enumerate(SELECTED_CLASSES):
            f.write(f"{class_name}\n")
    print(f"Labels written to {labels_path}")
    for idx, name in enumerate(SELECTED_CLASSES):
        print(f"  {idx}: {name}")

    # ------------------------------------------------------------------
    # Step 6 — Also save split metadata as JSON for downstream scripts
    # ------------------------------------------------------------------
    meta = {
        "image_size": list(IMAGE_SIZE),
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "classes": SELECTED_CLASSES,
        "stats": stats,
    }
    meta_path = PROCESSED_DIR / "dataset_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    # ------------------------------------------------------------------
    # Step 7 — Summary report
    # ------------------------------------------------------------------
    print("\n=== Split Summary ===")
    print(f"{'Class':<40} {'Total':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 66)
    for class_name in SELECTED_CLASSES:
        s = stats[class_name]
        print(
            f"{class_name:<40} {s['total']:>6} {s['train']:>6} "
            f"{s['val']:>6} {s['test']:>6}"
        )
    total_train = sum(s["train"] for s in stats.values())
    total_val   = sum(s["val"]   for s in stats.values())
    total_test  = sum(s["test"]  for s in stats.values())
    print("-" * 66)
    print(
        f"{'TOTAL':<40} {total_images:>6} {total_train:>6} "
        f"{total_val:>6} {total_test:>6}"
    )
    print("\nPhase 1 data preparation complete.")


if __name__ == "__main__":
    prepare_dataset()
