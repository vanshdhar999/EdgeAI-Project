"""
evaluate.py — Full evaluation of the trained plant disease detection model.

Loads the best available trained model and runs inference on the test set,
reporting overall accuracy, per-class precision/recall/F1, and saving a
confusion matrix plot to docs/confusion_matrix.png.

Usage:
    conda run -n pydl python3 src/evaluate.py
"""

import sys
from pathlib import Path

# Allow sibling imports (e.g., augmentation.py) when run from any working dir
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — do not call plt.show()
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix

from augmentation import normalize

# ---------------------------------------------------------------------------
# Module-level path constants (all relative to this file's location)
# ---------------------------------------------------------------------------

_SRC_DIR    = Path(__file__).resolve().parent
_PROJECT    = _SRC_DIR.parent
_MODELS_DIR = _PROJECT / "models"
_DATA_DIR   = _PROJECT / "data" / "processed" / "test"
_DOCS_DIR   = _PROJECT / "docs"
_LABELS_TXT = _MODELS_DIR / "labels.txt"

# Candidate model paths, tried in order.
# SavedModel format is excluded: Keras 3's load_model() does not support it
# (only .keras / .h5). The SavedModel is used only by quantize.py via
# tf.lite.TFLiteConverter, not for Python-level evaluation.
_MODEL_CANDIDATES = [
    _MODELS_DIR / "checkpoints" / "best_stage1.keras",
    _MODELS_DIR / "checkpoints" / "best_stage2.keras",
]

# Dataset parameters
_IMAGE_SIZE  = (224, 224)
_BATCH_SIZE  = 32

# Confusion matrix output
_CM_OUTPUT   = _DOCS_DIR / "confusion_matrix.png"
_CM_DPI      = 150
_CM_FIGSIZE  = (8, 6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(labels_path: Path) -> list[str]:
    """
    Read class names from labels.txt, one name per line.

    Args:
        labels_path: Path to labels.txt generated during training.

    Returns:
        Ordered list of class name strings matching the model's output indices.

    Raises:
        FileNotFoundError: If labels.txt does not exist.
    """
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.txt not found at {labels_path}. "
            "Run train.py first to generate the model and label file."
        )
    names = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    return names


def load_model(candidates: list[Path]) -> tuple[keras.Model, Path]:
    """
    Attempt to load a Keras model from each candidate path in order.

    Args:
        candidates: Ordered list of paths to try (SavedModel dirs or .keras files).

    Returns:
        (model, path) — the loaded Keras model and the path it was loaded from.

    Raises:
        FileNotFoundError: If none of the candidate paths exist.
    """
    for candidate in candidates:
        if candidate.exists():
            print(f"Loading model from: {candidate}")
            model = keras.models.load_model(str(candidate))
            return model, candidate

    paths_tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "No trained model found. Tried:\n  " + paths_tried + "\n"
        "Run train.py first to produce a saved model."
    )


def build_test_dataset(
    data_dir: Path,
    class_names: list[str],
    image_size: tuple[int, int],
    batch_size: int,
) -> tf.data.Dataset:
    """
    Build a normalised, non-shuffled tf.data pipeline over the test directory.

    Args:
        data_dir:    Root of the test split (contains one sub-folder per class).
        class_names: Ordered class names that determine label indices. Must
                     match the order used during training (from labels.txt).
        image_size:  (H, W) to resize images to.
        batch_size:  Number of images per batch.

    Returns:
        A tf.data.Dataset yielding (normalised_images, int_labels) batches.
    """
    ds = keras.utils.image_dataset_from_directory(
        str(data_dir),
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    actual_names = ds.class_names
    if actual_names != list(class_names):
        raise RuntimeError(
            f"Class order mismatch between labels.txt and test directory.\n"
            f"  labels.txt : {list(class_names)}\n"
            f"  test dir   : {actual_names}"
        )
    # Normalise: uint8 [0, 255] → float32 [0, 1]
    ds = ds.map(
        lambda images, labels: (normalize(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def run_inference(
    model: keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference over the entire dataset and collect true/predicted labels.

    Args:
        model:   Loaded Keras model with a softmax output.
        dataset: Batched, normalised tf.data.Dataset (shuffle=False).

    Returns:
        (y_true, y_pred) — 1-D integer arrays of true and predicted class indices.
    """
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for images, labels in dataset:
        predictions = model(images, training=False)          # [B, num_classes]
        pred_indices = np.argmax(predictions.numpy(), axis=1)
        all_true.append(labels.numpy())
        all_pred.append(pred_indices)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


def short_label(class_name: str) -> str:
    """
    Strip a well-known species prefix for more readable axis tick labels.

    E.g., "Tomato___Early_blight" → "Early_blight".
    If the name contains "___", only the suffix is returned.

    Args:
        class_name: Full class name string from labels.txt.

    Returns:
        Shortened display label.
    """
    if "___" in class_name:
        return class_name.split("___", maxsplit=1)[1]
    return class_name


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    figsize: tuple[int, int] = _CM_FIGSIZE,
    dpi: int = _CM_DPI,
) -> None:
    """
    Render and save a confusion matrix as a matplotlib figure.

    Cell counts are displayed as integers. Axes use shortened class name labels.
    The figure is saved to disk and the plot is NOT shown interactively.

    Args:
        cm:           Confusion matrix array, shape (n_classes, n_classes).
        class_names:  Ordered list of class name strings.
        output_path:  Destination .png path.
        figsize:      Matplotlib figure size in inches (width, height).
        dpi:          Output resolution in dots per inch.
    """
    short_names = [short_label(name) for name in class_names]
    n = len(short_names)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)

    # Annotate each cell with the integer count
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
    fig.savefig(str(output_path), dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate model loading, test-set evaluation, and result reporting.

    Prints overall accuracy, a per-class classification report, and saves a
    confusion matrix image to docs/confusion_matrix.png.
    """
    # --- Load class names ---
    class_names = load_class_names(_LABELS_TXT)
    num_classes = len(class_names)

    # --- Load model ---
    model, model_path = load_model(_MODEL_CANDIDATES)

    # --- Build test dataset ---
    test_ds = build_test_dataset(
        data_dir=_DATA_DIR,
        class_names=class_names,
        image_size=_IMAGE_SIZE,
        batch_size=_BATCH_SIZE,
    )

    # --- Run inference ---
    y_true, y_pred = run_inference(model, test_ds)

    # --- Compute metrics ---
    overall_accuracy = np.mean(y_true == y_pred) * 100.0
    missing_preds = [class_names[i] for i in range(len(class_names)) if i not in y_pred]
    if missing_preds:
        print(f"WARNING: {len(missing_preds)} class(es) have zero predictions: {missing_preds}")

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # --- Print results ---
    print()
    print("=== Evaluation Results ===")
    print(f"Model loaded from: {model_path}")
    print(f"Test images: {len(y_true)}")
    print()
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    print()
    print("=== Per-Class Report ===")
    print(report)

    # --- Save confusion matrix ---
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=_CM_OUTPUT,
    )
    print(f"=== Confusion Matrix saved to {_CM_OUTPUT} ===")


if __name__ == "__main__":
    main()
