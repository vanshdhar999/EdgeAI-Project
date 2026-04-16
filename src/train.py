"""
train.py — Two-stage fine-tuning of MobileNetV3Small for plant disease detection.

Stage 1: Freeze the base model, train only the classification head (10 epochs).
Stage 2: Unfreeze the last 30 layers of the base model and fine-tune (15 epochs).

After both stages, the best checkpoint is loaded and exported as a SavedModel
for downstream TFLite conversion (Phase 3).

Usage:
    conda run -n pydl python3 src/train.py
"""

import shutil
import sys
from pathlib import Path

# Allow `from augmentation import ...` when running from the project root.
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
import keras
from keras import layers

from augmentation import normalize, build_augmentation_pipeline


# ---------------------------------------------------------------------------
# Config — all hyper-parameters in one place for easy tuning
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 32
IMAGE_SIZE: tuple[int, int] = (224, 224)

STAGE1_EPOCHS: int = 10
STAGE2_EPOCHS: int = 15

STAGE1_LR: float = 1e-3
STAGE2_LR: float = 1e-5

STAGE2_UNFREEZE_LAYERS: int = 30

RANDOM_SEED: int = 42

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
LABELS_FILE: Path = PROJECT_ROOT / "models" / "labels.txt"
CHECKPOINT_DIR: Path = PROJECT_ROOT / "models" / "checkpoints"
BEST_CHECKPOINT_STAGE1: Path = CHECKPOINT_DIR / "best_stage1.keras"
BEST_CHECKPOINT_STAGE2: Path = CHECKPOINT_DIR / "best_stage2.keras"
SAVEDMODEL_DIR: Path = PROJECT_ROOT / "models" / "plant_disease_savedmodel"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(labels_path: Path) -> list[str]:
    """
    Read class names from labels.txt, one name per line, preserving order.

    Args:
        labels_path: Path to the labels file.

    Returns:
        Ordered list of class name strings.
    """
    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_dataset(
    split_dir: Path,
    class_names: list[str],
    augment: bool = False,
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Build a tf.data pipeline for a given split directory.

    Images are loaded from `split_dir/<class>/` subdirectories.  Normalisation
    (pixel / 255.0) is applied to every split.  Augmentation is applied only
    when ``augment=True`` (training set).

    No .cache() is used because the processed dataset (~227 MB) fits the no-cache
    constraint from the project spec.

    Args:
        split_dir:   Root of the split, e.g. data/processed/train/.
        class_names: Ordered list of class names matching labels.txt.
        augment:     If True, apply the training augmentation pipeline.

    Returns:
        Tuple of (dataset, actual_class_names) where:
        - dataset is a batched, prefetched tf.data.Dataset yielding (image, label)
          pairs where image is float32 [0, 1] and label is an int32 class index.
        - actual_class_names is the alphabetically sorted list Keras actually used,
          read from the raw dataset before transformations are applied.
          (.class_names is lost after .map()/.prefetch() wrap the dataset.)
    """
    raw_ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        str(split_dir),
        class_names=class_names,   # restricts which classes to include
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=(augment),         # shuffle only the training set
        seed=RANDOM_SEED,
    )

    # Capture class names NOW — before chaining .map()/.prefetch() which
    # wrap the dataset in _MapDataset/_PrefetchDataset and lose this attribute.
    actual_class_names: list[str] = raw_ds.class_names

    ds = raw_ds

    # Normalise: uint8 [0,255] → float32 [0,1]
    ds = ds.map(
        lambda x, y: (normalize(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Augment training images element-wise inside each batch
    if augment:
        augment_fn = build_augmentation_pipeline()
        ds = ds.map(
            lambda x, y: (tf.map_fn(augment_fn, x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, actual_class_names


def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    """
    Construct the full model from a pretrained MobileNetV3Small base plus a
    custom classification head.

    Args:
        num_classes: Number of output disease classes.

    Returns:
        (model, base_model) — the full compiled model and a reference to the
        base so Stage 2 can selectively unfreeze layers.
    """
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,  # we do our own pixel/255 normalisation
    )
    base_model.trainable = False  # frozen for Stage 1

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    # training=False keeps BatchNorm layers in inference mode during both stages.
    # This preserves pretrained running statistics, which is the recommended
    # approach for fine-tuning on small datasets.
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="plant_disease_mobilenetv3")
    return model, base_model


def make_callbacks(stage: int, checkpoint_path: Path) -> list[keras.callbacks.Callback]:
    """
    Return the standard callback set for a given training stage.

    Each stage writes its best checkpoint to a separate file so that the
    globally best checkpoint across both stages can be selected after training
    (see Fix 3).  Sharing a single path would allow Stage 2 to overwrite a
    superior Stage 1 checkpoint if Stage 2 never improves on it.

    Args:
        stage:           Integer stage number (1 or 2), used to name the CSV
                         log file.
        checkpoint_path: Path where this stage's best checkpoint is saved.

    Returns:
        List of Keras callbacks.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    early_stop_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )
    csv_logger_cb = keras.callbacks.CSVLogger(
        str(CHECKPOINT_DIR / f"training_stage{stage}.csv"),
        append=False,
    )

    return [checkpoint_cb, early_stop_cb, reduce_lr_cb, csv_logger_cb]


def print_stage_summary(
    stage: int,
    history: keras.callbacks.History,
    checkpoint_path: Path,
) -> None:
    """
    Print a concise summary after each training stage.

    Args:
        stage:           Stage number (1 or 2).
        history:         Keras History object from model.fit().
        checkpoint_path: Path where the best checkpoint was saved.
    """
    epochs_run = len(history.history["val_accuracy"])
    best_val_acc = max(history.history["val_accuracy"])
    print(
        f"\n{'='*60}\n"
        f"  Stage {stage} complete\n"
        f"  Epochs run        : {epochs_run}\n"
        f"  Best val_accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)\n"
        f"  Best checkpoint   : {checkpoint_path}\n"
        f"{'='*60}\n"
    )


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    """
    Run the full two-stage training pipeline and export the final SavedModel.

    Stage 1 — Head only (base frozen):
        Compile with Adam(lr=1e-3).  Train for up to STAGE1_EPOCHS epochs.

    Stage 2 — Partial unfreeze (last STAGE2_UNFREEZE_LAYERS of base):
        Recompile with Adam(lr=1e-5).  Train for up to STAGE2_EPOCHS epochs.

    After training, load the best checkpoint and export to SavedModel format
    for Phase 3 TFLite conversion.
    """
    tf.random.set_seed(RANDOM_SEED)

    # ------------------------------------------------------------------
    # 1. Load class names and build datasets
    # ------------------------------------------------------------------
    class_names = load_class_names(LABELS_FILE)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}) from labels.txt: {class_names}")

    # keras.utils.image_dataset_from_directory always sorts class_names
    # alphabetically regardless of the order passed in.  build_dataset() reads
    # back the actual sorted order from the raw dataset (before .map/.prefetch
    # wrap it and lose .class_names) and returns it alongside the dataset.
    train_ds, actual_classes = build_dataset(DATA_DIR / "train", class_names, augment=True)

    LABELS_FILE.write_text("\n".join(actual_classes) + "\n", encoding="utf-8")
    print(f"labels.txt updated to match training order: {actual_classes}")

    # Val dataset must use the same sorted class list so all splits share
    # identical label-to-index mappings.
    val_ds, _ = build_dataset(DATA_DIR / "val", actual_classes, augment=False)

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    model, base_model = build_model(num_classes)
    model.summary()

    # ------------------------------------------------------------------
    # 3. Stage 1 — train head only
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  STAGE 1: Training classification head (base frozen)")
    print("="*60 + "\n")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=make_callbacks(stage=1, checkpoint_path=BEST_CHECKPOINT_STAGE1),
    )

    print_stage_summary(
        stage=1,
        history=history_stage1,
        checkpoint_path=BEST_CHECKPOINT_STAGE1,
    )

    # ------------------------------------------------------------------
    # 4. Stage 2 — unfreeze last N layers and fine-tune
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  STAGE 2: Fine-tuning last {STAGE2_UNFREEZE_LAYERS} base layers")
    print("="*60 + "\n")

    base_model.trainable = True
    # Freeze all layers except the last STAGE2_UNFREEZE_LAYERS
    for layer in base_model.layers[:-STAGE2_UNFREEZE_LAYERS]:
        layer.trainable = False

    # Recompile with a much lower learning rate to avoid destroying pretrained weights
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=STAGE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=make_callbacks(stage=2, checkpoint_path=BEST_CHECKPOINT_STAGE2),
    )

    print_stage_summary(
        stage=2,
        history=history_stage2,
        checkpoint_path=BEST_CHECKPOINT_STAGE2,
    )

    # ------------------------------------------------------------------
    # 5. Load best checkpoint and export as SavedModel
    # ------------------------------------------------------------------
    # Compare best val_accuracy from each stage and load whichever checkpoint
    # achieved the higher value.  Using per-stage checkpoint files (Fix 3)
    # prevents Stage 2 from silently overwriting a superior Stage 1 checkpoint.
    best_val_acc_stage1 = max(history_stage1.history["val_accuracy"])
    best_val_acc_stage2 = max(history_stage2.history["val_accuracy"])

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

    print(f"Loading best checkpoint for final export: {best_checkpoint}")
    best_model = keras.models.load_model(str(best_checkpoint))

    # Remove any stale SavedModel directory before export to avoid mixing
    # artefacts from different training runs.
    if SAVEDMODEL_DIR.exists():
        shutil.rmtree(SAVEDMODEL_DIR)
    # Use tf.saved_model.save() instead of model.export(): Keras 3's export()
    # goes through ExportArchive which calls inspect.getattr_static() on a
    # TF _DictWrapper — that fails on Python 3.12. tf.saved_model.save() uses
    # a completely different code path and produces an identical TF SavedModel
    # that TFLiteConverter.from_saved_model() can consume in Phase 3.
    tf.saved_model.save(best_model, str(SAVEDMODEL_DIR))

    print(
        f"\nSavedModel exported to: {SAVEDMODEL_DIR}\n"
        "Ready for Phase 3 TFLite conversion.\n"
    )

    # Test-set evaluation (confusion matrix, per-class precision/recall) is
    # handled by src/evaluate.py — run it after training completes.


if __name__ == "__main__":
    train()
