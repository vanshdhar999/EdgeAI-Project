"""
data_prep.py — PlantVillage dataset preparation for Edge AI plant disease detection.

Supports four dataset modes selectable via DATASET_MODE:

  "tomato_5"    — 5 tomato classes (quick pipeline validation)
  "tomato_10"   — All 10 tomato classes (Option 1)
  "multi_crop"  — 15 classes: Tomato (10) + Potato (3) + Pepper (2) (Option 2)
  "balanced"    — 10 classes across 4 crops, capped at MAX_IMAGES_PER_CLASS
                  Tomato (3) + Potato (2) + Pepper (2) + Corn (3)
                  Equal class distribution — recommended for best model fairness

Set DATASET_MODE below, then run:
    conda run -n pydl python3 src/data_prep.py

Outputs:
    data/processed/train/<class>/   — training images
    data/processed/val/<class>/     — validation images
    data/processed/test/<class>/    — test images
    models/labels.txt               — class index to disease name mapping
    data/processed/dataset_meta.json
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# *** CHANGE THIS TO SWITCH DATASET MODE ***
# ---------------------------------------------------------------------------
# Options:
#   "tomato_5"    — 5 tomato classes  (quick pipeline validation)
#   "tomato_10"   — 10 tomato classes (Option 1)
#   "multi_crop"  — 15 classes: Tomato + Potato + Pepper (Option 2)
#   "balanced"    — 10 classes, 4 crops, equal distribution (recommended)
# ---------------------------------------------------------------------------

DATASET_MODE: str = "balanced"

# Maximum images per class for the "balanced" mode.
# Classes with more images are randomly downsampled to this cap so every
# class contributes equally to training.
# Set to None to disable capping (use all available images).
MAX_IMAGES_PER_CLASS: int | None = 1000


# ---------------------------------------------------------------------------
# Class definitions per mode
# ---------------------------------------------------------------------------

_TOMATO_5 = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Bacterial_spot",
    "Tomato___Septoria_leaf_spot",
]

_TOMATO_10 = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
]

_MULTI_CROP = _TOMATO_10 + [
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",
]

# Balanced: 4 crops, 10 classes, ~1000 images each.
# Selection criteria:
#   - Each crop has at least 1 healthy + 1 disease class
#   - All selected classes have >= 950 raw images (Potato___healthy excluded —
#     only 152 images, too small for meaningful training)
#   - Capped at MAX_IMAGES_PER_CLASS to equalise large classes
#     (e.g. Tomato___Late_blight=1908, Pepper___healthy=1478)
_BALANCED = [
    # Tomato — 3 classes (removed 7 classes to reduce tomato dominance)
    "Tomato___Early_blight",       # 1000 raw
    "Tomato___Late_blight",        # 1908 raw → capped
    "Tomato___healthy",            # 1591 raw → capped
    # Potato — 2 classes (healthy excluded: only 152 raw images)
    "Potato___Early_blight",       # 1000 raw
    "Potato___Late_blight",        # 1000 raw
    # Pepper — 2 classes
    "Pepper,_bell___Bacterial_spot",  # 997 raw
    "Pepper,_bell___healthy",         # 1478 raw → capped
    # Corn — 3 classes
    "Corn_(maize)___Common_rust_",         # 1192 raw → capped
    "Corn_(maize)___Northern_Leaf_Blight", # 985 raw
    "Corn_(maize)___healthy",              # 1162 raw → capped
]

CLASS_SETS: dict[str, list[str]] = {
    "tomato_5":   _TOMATO_5,
    "tomato_10":  _TOMATO_10,
    "multi_crop": _MULTI_CROP,
    "balanced":   _BALANCED,
}

if DATASET_MODE not in CLASS_SETS:
    raise ValueError(
        f"Unknown DATASET_MODE '{DATASET_MODE}'. "
        f"Choose from: {list(CLASS_SETS.keys())}"
    )

SELECTED_CLASSES: list[str] = CLASS_SETS[DATASET_MODE]

# Only apply the cap in balanced mode (other modes use all images)
_EFFECTIVE_CAP: int | None = (
    MAX_IMAGES_PER_CLASS if DATASET_MODE == "balanced" else None
)


# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

RAW_DATASET_DIR = Path("plantvillage-dataset/color")
PROCESSED_DIR   = Path("data/processed")
MODELS_DIR      = Path("models")

IMAGE_SIZE   = (224, 224)
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_images(class_dir: Path, cap: int | None, seed: int) -> list[Path]:
    """
    Return a sorted list of valid image paths inside class_dir.
    If cap is set and the class has more images, randomly sample down to cap.
    """
    all_paths = sorted(p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS)
    if cap is not None and len(all_paths) > cap:
        rng = random.Random(seed)
        all_paths = rng.sample(all_paths, cap)
        all_paths = sorted(all_paths)
    return all_paths


def stratified_split(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


def resize_and_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
        img.save(dst.with_suffix(".jpg"), format="JPEG", quality=95)


def clear_processed_if_mode_changed() -> None:
    """Delete processed data if the class set or cap changed since last run."""
    meta_path = PROCESSED_DIR / "dataset_meta.json"
    if not meta_path.exists():
        return

    with meta_path.open() as f:
        meta = json.load(f)

    old_classes = set(meta.get("classes", []))
    old_cap     = meta.get("max_images_per_class")
    new_classes = set(SELECTED_CLASSES)
    new_cap     = _EFFECTIVE_CAP

    if old_classes != new_classes or old_cap != new_cap:
        print(
            f"  Dataset changed ({meta.get('mode', '?')} → {DATASET_MODE}, "
            f"cap {old_cap} → {new_cap}). Clearing old processed data..."
        )
        shutil.rmtree(PROCESSED_DIR)
        print(f"  Cleared {PROCESSED_DIR}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_dataset() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=== Phase 1: Data Preparation ===\n")
    print(f"  Dataset mode         : {DATASET_MODE}")
    print(f"  Classes              : {len(SELECTED_CLASSES)}")
    print(f"  Max images per class : {_EFFECTIVE_CAP if _EFFECTIVE_CAP else 'unlimited'}")
    print(f"  Raw data             : {RAW_DATASET_DIR.resolve()}")
    print(f"  Image size           : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"  Split                : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}\n")

    # ------------------------------------------------------------------
    # Step 1 — Clear stale processed data if mode/cap changed
    # ------------------------------------------------------------------
    clear_processed_if_mode_changed()

    # ------------------------------------------------------------------
    # Step 2 — Validate class directories
    # ------------------------------------------------------------------
    print("Validating class directories...")
    missing = [c for c in SELECTED_CLASSES if not (RAW_DATASET_DIR / c).exists()]
    if missing:
        available = sorted(d.name for d in RAW_DATASET_DIR.iterdir() if d.is_dir())
        raise FileNotFoundError(
            f"Missing {len(missing)} class director(ies) in {RAW_DATASET_DIR}:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nAvailable:\n"
            + "\n".join(f"  {a}" for a in available)
        )
    print(f"  [OK] All {len(SELECTED_CLASSES)} class directories found.\n")

    # ------------------------------------------------------------------
    # Step 3 — Collect (and optionally cap) images, then split
    # ------------------------------------------------------------------
    split_registry: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    stats: dict[str, dict[str, int]] = {}

    for class_name in SELECTED_CLASSES:
        all_images = collect_images(
            RAW_DATASET_DIR / class_name,
            cap=_EFFECTIVE_CAP,
            seed=RANDOM_SEED,
        )
        if not all_images:
            raise ValueError(f"No images found in {RAW_DATASET_DIR / class_name}")

        train, val, test = stratified_split(
            all_images, TRAIN_RATIO, VAL_RATIO, seed=RANDOM_SEED
        )
        split_registry["train"][class_name] = train
        split_registry["val"][class_name]   = val
        split_registry["test"][class_name]  = test
        stats[class_name] = {
            "total": len(all_images),
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        }

    # ------------------------------------------------------------------
    # Step 4 — Verify no data leakage
    # ------------------------------------------------------------------
    print("Verifying no data leakage across splits...")
    for class_name in SELECTED_CLASSES:
        train_names = {p.stem for p in split_registry["train"][class_name]}
        val_names   = {p.stem for p in split_registry["val"][class_name]}
        test_names  = {p.stem for p in split_registry["test"][class_name]}
        overlaps = {
            "train∩val":  train_names & val_names,
            "train∩test": train_names & test_names,
            "val∩test":   val_names   & test_names,
        }
        if any(overlaps.values()):
            raise AssertionError(
                f"Data leakage in '{class_name}':\n"
                + "\n".join(f"  {k}: {v}" for k, v in overlaps.items() if v)
            )
    print("  [OK] No leakage detected.\n")

    # ------------------------------------------------------------------
    # Step 5 — Resize and copy
    # ------------------------------------------------------------------
    total_images = sum(s["total"] for s in stats.values())
    processed    = 0

    print(f"Copying and resizing {total_images} images...")
    for split_name in ("train", "val", "test"):
        for class_name, paths in split_registry[split_name].items():
            for src in paths:
                dst = PROCESSED_DIR / split_name / class_name / src.name
                resize_and_copy(src, dst)
                processed += 1
                if processed % 500 == 0:
                    print(f"  {processed}/{total_images} images processed...")

    print(f"  {processed}/{total_images} images — done.\n")

    # ------------------------------------------------------------------
    # Step 6 — Write labels.txt and metadata
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    labels_path = MODELS_DIR / "labels.txt"
    labels_path.write_text("\n".join(SELECTED_CLASSES) + "\n", encoding="utf-8")

    print(f"Labels written to {labels_path}:")
    for idx, name in enumerate(SELECTED_CLASSES):
        print(f"  {idx:>2}: {name}")

    meta = {
        "mode":                 DATASET_MODE,
        "max_images_per_class": _EFFECTIVE_CAP,
        "image_size":           list(IMAGE_SIZE),
        "train_ratio":          TRAIN_RATIO,
        "val_ratio":            VAL_RATIO,
        "test_ratio":           TEST_RATIO,
        "random_seed":          RANDOM_SEED,
        "classes":              SELECTED_CLASSES,
        "stats":                stats,
    }
    meta_path = PROCESSED_DIR / "dataset_meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    # ------------------------------------------------------------------
    # Step 7 — Summary
    # ------------------------------------------------------------------
    print("\n=== Split Summary ===")
    print(f"{'Class':<52} {'Raw':>6} {'Used':>6} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 90)

    current_crop = ""
    for class_name in SELECTED_CLASSES:
        crop = class_name.split("___")[0]
        if crop != current_crop:
            current_crop = crop
            print(f"  [{crop}]")

        s        = stats[class_name]
        raw_count = sum(
            1 for p in (RAW_DATASET_DIR / class_name).iterdir()
            if p.suffix in IMAGE_EXTENSIONS
        )
        capped = " ↓" if _EFFECTIVE_CAP and raw_count > s["total"] else "  "
        short  = class_name.split("___")[1] if "___" in class_name else class_name
        print(
            f"  {short:<50}{capped} {raw_count:>6} {s['total']:>6} "
            f"{s['train']:>6} {s['val']:>6} {s['test']:>6}"
        )

    total_used  = sum(s["total"] for s in stats.values())
    total_train = sum(s["train"] for s in stats.values())
    total_val   = sum(s["val"]   for s in stats.values())
    total_test  = sum(s["test"]  for s in stats.values())
    print("-" * 90)
    print(
        f"  {'TOTAL':<52} {total_used:>6} {total_used:>6} "
        f"{total_train:>6} {total_val:>6} {total_test:>6}"
    )
    print(f"\nPhase 1 complete — mode: {DATASET_MODE}, {len(SELECTED_CLASSES)} classes.\n")


if __name__ == "__main__":
    prepare_dataset()
