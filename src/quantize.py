"""
quantize.py — INT8 post-training quantization and TFLite export.

Converts the trained .keras checkpoint to two TFLite variants:
  1. Float32 baseline  → models/plant_disease_float32.tflite
  2. INT8 quantized    → models/plant_disease.tflite  (final deployment model)

Conversion path: .keras checkpoint → temporary SavedModel → TFLite.
TFLiteConverter.from_saved_model() is the canonical conversion path and
correctly names all tensors so calibration statistics can be matched to
the right ops. from_keras_model() was removed: it relies on a Keras 2
internal hook (keras_deps.get_call_context_function) that Keras 3 removed.
from_concrete_functions() was removed: it produces tensors with generated
names that TFLite's calibrator cannot match, causing silent calibration
failure and completely wrong INT8 activation ranges.

INT8 quantization uses a representative dataset of 200 training-set images to
calibrate activation ranges.  Float32 I/O is preserved so inference.py and
live_camera.py can pass normalised float32 images ([0, 1]) without extra
preprocessing steps.

Usage (on GPU training machine):
    python3 src/quantize.py

Outputs:
    models/plant_disease_float32.tflite  — unquantized baseline
    models/plant_disease.tflite          — INT8 quantized (deploy this)
    models/labels.txt                    — class names (already present)

Requirements:
    tensorflow>=2.18.0  (same env used for training)
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import tensorflow as tf
import keras

# ---------------------------------------------------------------------------
# Compatibility patches: Python 3.12 + Keras 3 + TF 2.16
# ---------------------------------------------------------------------------
def _apply_compat_patches() -> None:
    """
    Two targeted patches for Python 3.12 + Keras 3 running against TF 2.16.

    Patch 1 — inspect._check_instance (Python 3.12 only):
      Python 3.12 tightened inspect.getattr_static() so it raises TypeError on
      objects whose __dict__ descriptor is non-standard (e.g. TF's _DictWrapper).
      This propagates through typing.__instancecheck__ and breaks isinstance()
      checks inside TF. Patching _check_instance to return {} on TypeError
      lets isinstance() complete normally without skipping any serialization logic.

    Patch 2 — Converter.convert_to_trackable (Keras 3 + TF 2.16):
      Keras 3 stores some model attributes as plain Python dicts rather than
      the Keras 2 DictWrapper (a Trackable subclass). When tf.saved_model.save()
      and model.export() traverse the Trackable object graph they call
      converter.convert_to_trackable(ref) on each child. If ref is a plain dict
      the function reaches `obj.dtype not in (...)` and crashes:
        AttributeError: 'dict' object has no attribute 'dtype'
      Returning None for any ref that raises AttributeError correctly signals
      "not a trackable object, skip it". Actual trainable variables are always
      reached through the proper Trackable Layer hierarchy, so nothing is lost.
    """
    import sys
    import inspect

    # Patch 1 — Python 3.12 + TF inspect issue
    if sys.version_info >= (3, 12):
        try:
            _orig_check = inspect._check_instance

            def _safe_check_instance(obj, attr):
                try:
                    return _orig_check(obj, attr)
                except TypeError:
                    return {}

            inspect._check_instance = _safe_check_instance
        except Exception:
            pass

    # Patch 2 — Keras 3 dict attributes in Trackable graph traversal
    try:
        from tensorflow.python.trackable import converter as _tconv

        _orig_ctt = _tconv.Converter.convert_to_trackable

        def _safe_convert_to_trackable(self, ref, parent=None):
            try:
                return _orig_ctt(self, ref, parent=parent)
            except AttributeError:
                return None  # Skip plain dicts and other non-Tensor/Trackable objects

        _tconv.Converter.convert_to_trackable = _safe_convert_to_trackable
    except Exception:
        pass


_apply_compat_patches()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
SAVEDMODEL_DIR: Path = MODELS_DIR / "plant_disease_savedmodel"
CHECKPOINT_DIR: Path = MODELS_DIR / "checkpoints"
LABELS_FILE: Path = MODELS_DIR / "labels.txt"
DATA_TRAIN_DIR: Path = PROJECT_ROOT / "data" / "processed" / "train"

# Output TFLite paths
TFLITE_FLOAT32: Path = MODELS_DIR / "plant_disease_float32.tflite"
TFLITE_INT8: Path = MODELS_DIR / "plant_disease.tflite"

# Number of representative samples for INT8 calibration
NUM_CALIBRATION_SAMPLES: int = 200

# Model input shape expected by MobileNetV3Small
IMAGE_SIZE: tuple[int, int] = (224, 224)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(labels_path: Path) -> list[str]:
    """Return ordered class names from labels.txt."""
    return [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_source_model() -> tuple[keras.Model, str]:
    """
    Load the trained Keras model from a .keras checkpoint.

    Returns:
        (keras_model, source_description)
    """
    for ckpt in (CHECKPOINT_DIR / "best_stage1.keras",
                 CHECKPOINT_DIR / "best_stage2.keras"):
        if ckpt.exists():
            print(f"  Loading Keras checkpoint: {ckpt}")
            model = keras.models.load_model(str(ckpt))
            return model, f"Keras checkpoint ({ckpt.name})"

    raise FileNotFoundError(
        "No .keras checkpoint found. Run src/train.py first.\n"
        f"  Looked for: {CHECKPOINT_DIR / 'best_stage1.keras'}\n"
        f"             {CHECKPOINT_DIR / 'best_stage2.keras'}"
    )


def collect_calibration_images(num_samples: int = NUM_CALIBRATION_SAMPLES) -> list[np.ndarray]:
    """
    Collect a balanced sample of normalised training images for INT8 calibration.

    Samples are drawn proportionally from each class directory so that all
    disease categories contribute to the activation calibration.

    Args:
        num_samples: Total number of images to collect.

    Returns:
        List of float32 numpy arrays, each shaped (224, 224, 3) in [0, 1].
    """
    from PIL import Image

    class_dirs = sorted(
        d for d in DATA_TRAIN_DIR.iterdir() if d.is_dir()
    )
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subdirectories found in {DATA_TRAIN_DIR}. "
            "Run src/data_prep.py first."
        )

    per_class = max(1, num_samples // len(class_dirs))
    images: list[np.ndarray] = []

    for cls_dir in class_dirs:
        img_paths = sorted(cls_dir.glob("*.jpg"))[:per_class]
        for path in img_paths:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
                arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
                images.append(arr)

    print(f"  Calibration images collected: {len(images)} "
          f"(~{len(images) // len(class_dirs)} per class across {len(class_dirs)} classes)")
    return images


def make_representative_dataset(calibration_images: list[np.ndarray]):
    """
    Return a generator function compatible with TFLiteConverter.representative_dataset.

    Each call yields a list containing one float32 tensor of shape [1, 224, 224, 3].
    """
    def generator():
        for img in calibration_images:
            yield [img[np.newaxis, ...].astype(np.float32)]

    return generator


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def save_temp_savedmodel(source: keras.Model) -> Path:
    """
    Serialize the Keras model to a temporary SavedModel directory via Keras 3's
    model.export() rather than tf.saved_model.save().

    Why not tf.saved_model.save(model, ...):
      Keras 3 stores some internal attributes as plain Python dicts. When
      tf.saved_model.save() traverses the Trackable object graph it reaches
      one of these dicts and calls obj.dtype on it, crashing with:
        AttributeError: 'dict' object has no attribute 'dtype'

    Why model.export() works:
      Keras 3's export() wraps the model in an ExportArchive — a clean
      tf.Module whose Trackable graph is built solely from traced concrete
      functions and explicitly tracked variables. tf.saved_model.save() is
      then called on that tf.Module (not the raw Keras model), so it never
      encounters plain dict attributes.

    Why from_saved_model() is used for TFLite conversion:
      It is the canonical path and assigns stable tensor names to all ops.
      TFLite's calibrator matches representative dataset samples to input
      tensors by name; stable names are required for correct INT8 calibration.
      from_concrete_functions() produces auto-generated names that the
      calibrator cannot match, causing silent zero-statistics and wrong
      INT8 activation ranges.

    Returns:
        Path to the temporary SavedModel directory (caller must delete it).
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="tflite_export_"))
    source.export(str(tmp_dir))
    return tmp_dir


def convert_to_tflite_float32(savedmodel_dir: Path) -> bytes:
    """
    Convert a SavedModel to a float32 TFLite model (unquantized baseline).

    Args:
        savedmodel_dir: Path to a SavedModel directory produced by save_temp_savedmodel().

    Returns:
        Serialised TFLite flatbuffer bytes.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))
    return converter.convert()


def convert_to_tflite_int8(
    savedmodel_dir: Path,
    calibration_images: list[np.ndarray],
) -> bytes:
    """
    Convert a SavedModel to an INT8 post-training quantized TFLite model.

    Quantization strategy:
    - Weights and activations are quantized to INT8.
    - Input and output tensors remain float32 so that inference code can pass
      normalised [0, 1] images without additional scale/zero-point arithmetic.
    - Calibration uses NUM_CALIBRATION_SAMPLES images from the training set to
      determine per-layer activation ranges.

    Args:
        savedmodel_dir:     Path to a SavedModel directory.
        calibration_images: Pre-collected calibration images (float32, [0, 1]).

    Returns:
        Serialised TFLite flatbuffer bytes.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(str(savedmodel_dir))

    # Enable post-training integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(calibration_images)

    # Target: all ops in INT8 on TFLite built-ins
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Keep float32 I/O so inference.py doesn't need scale/zero-point handling
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    return converter.convert()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_outputs(
    source: keras.Model,
    tflite_bytes: bytes,
    calibration_images: list[np.ndarray],
    num_checks: int = 10,
    tolerance: float = 0.05,
) -> bool:
    """
    Verify that TFLite model outputs closely match the Keras model outputs
    on the same inputs.

    Args:
        source:             Loaded keras.Model (reference for comparison).
        tflite_bytes:       INT8 TFLite model bytes.
        calibration_images: Pool of images to sample from.
        num_checks:         Number of images to compare.
        tolerance:          Max allowed absolute difference between top-1 probabilities.

    Returns:
        True if all checks pass, False otherwise.
    """
    def ref_predict(x: np.ndarray) -> np.ndarray:
        return source.predict(x, verbose=0)

    # Set up TFLite interpreter
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    mismatches = 0
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(calibration_images), size=min(num_checks, len(calibration_images)), replace=False)

    for i in indices:
        img = calibration_images[i][np.newaxis, ...].astype(np.float32)

        # Reference
        ref_out = ref_predict(img)
        ref_top1 = int(np.argmax(ref_out[0]))
        ref_conf = float(ref_out[0][ref_top1])

        # TFLite
        interp.set_tensor(inp_details[0]["index"], img)
        interp.invoke()
        tfl_out = interp.get_tensor(out_details[0]["index"])
        tfl_top1 = int(np.argmax(tfl_out[0]))
        tfl_conf = float(tfl_out[0][tfl_top1])

        match = ref_top1 == tfl_top1
        diff = abs(ref_conf - tfl_conf)
        status = "✓" if match and diff <= tolerance else "✗"
        print(f"  [{status}] sample {i:3d}: ref={ref_top1} ({ref_conf:.3f}) "
              f"tfl={tfl_top1} ({tfl_conf:.3f}) Δconf={diff:.4f}")

        if not (match and diff <= tolerance):
            mismatches += 1

    passed = num_checks - mismatches
    print(f"\n  Validation: {passed}/{num_checks} checks passed "
          f"(tolerance={tolerance:.2f})")
    return mismatches == 0


# ---------------------------------------------------------------------------
# Accuracy comparison on test set
# ---------------------------------------------------------------------------

def evaluate_tflite_accuracy(
    tflite_bytes: bytes,
    split: str = "test",
    max_images: int = 500,
) -> tuple[float, list[str]]:
    """
    Run the TFLite model on a subset of the test split and return top-1 accuracy.

    Class ordering is derived from the split directory (alphabetical sort),
    which matches image_dataset_from_directory's default ordering used during
    training. Using labels.txt ordering would be wrong if the file was written
    in a different order than the model's output indices.

    Args:
        tflite_bytes: Serialised TFLite model.
        split:        Which data split to evaluate on ("test" or "val").
        max_images:   Cap total images to keep evaluation fast on CPU.

    Returns:
        (top-1 accuracy in [0, 1], alphabetically-sorted class names used)
    """
    from PIL import Image

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    split_dir = PROJECT_ROOT / "data" / "processed" / split

    # Alphabetical sort — matches keras image_dataset_from_directory default
    class_names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())

    correct = 0
    total = 0
    per_class_limit = max(1, max_images // len(class_names))

    for class_idx, class_name in enumerate(class_names):
        cls_dir = split_dir / class_name
        img_paths = sorted(cls_dir.glob("*.jpg"))

        for path in img_paths[:per_class_limit]:
            with Image.open(path) as img:
                img_arr = np.array(
                    img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS),
                    dtype=np.float32,
                ) / 255.0

            inp = img_arr[np.newaxis, ...]
            interp.set_tensor(inp_details[0]["index"], inp)
            interp.invoke()
            pred = int(np.argmax(interp.get_tensor(out_details[0]["index"])[0]))

            if pred == class_idx:
                correct += 1
            total += 1

    return (correct / total if total > 0 else 0.0), class_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def quantize() -> None:
    """
    Full Phase 3 quantization pipeline:
      1. Load trained model and export to a temporary SavedModel.
      2. Convert to float32 TFLite (baseline) and evaluate accuracy.
      3. Collect calibration images from training set.
      4. Convert to INT8 TFLite with PTQ and evaluate accuracy.
      5. Validate INT8 outputs match reference model.
      6. Print size and accuracy comparison summary.
    """
    print("=== Phase 3: TFLite Quantization ===\n")

    class_names = load_class_names(LABELS_FILE)
    print(f"Classes from labels.txt ({len(class_names)}): {class_names}\n")

    # ------------------------------------------------------------------
    # 1. Load source model + export to temporary SavedModel
    # ------------------------------------------------------------------
    print("Step 1: Loading trained model and exporting to SavedModel...")
    source, source_desc = load_source_model()
    print(f"  Source: {source_desc}")
    print("  Saving to temporary SavedModel for TFLite conversion...")
    tmp_savedmodel = save_temp_savedmodel(source)
    print(f"  SavedModel written to: {tmp_savedmodel}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # ------------------------------------------------------------------
        # 2. Float32 TFLite baseline + accuracy
        # ------------------------------------------------------------------
        print("Step 2: Converting to float32 TFLite (baseline)...")
        t0 = time.time()
        float32_bytes = convert_to_tflite_float32(tmp_savedmodel)
        TFLITE_FLOAT32.write_bytes(float32_bytes)
        float32_size_mb = len(float32_bytes) / 1024 / 1024
        print(f"  Saved: {TFLITE_FLOAT32}  ({float32_size_mb:.2f} MB)  "
              f"[{time.time()-t0:.1f}s]")
        print("  Evaluating float32 accuracy on test split...")
        float32_acc, eval_class_names = evaluate_tflite_accuracy(float32_bytes)
        print(f"  Float32 test accuracy: {float32_acc*100:.2f}%")
        print(f"  (Class order used for eval: {eval_class_names})\n")

        # ------------------------------------------------------------------
        # 3. Collect calibration images
        # ------------------------------------------------------------------
        print("Step 3: Collecting calibration images...")
        calibration_images = collect_calibration_images(NUM_CALIBRATION_SAMPLES)
        print()

        # ------------------------------------------------------------------
        # 4. INT8 TFLite quantization + accuracy
        # ------------------------------------------------------------------
        print("Step 4: Converting to INT8 TFLite (post-training quantization)...")
        print("  This may take a few minutes on CPU...")
        t0 = time.time()
        int8_bytes = convert_to_tflite_int8(tmp_savedmodel, calibration_images)
        TFLITE_INT8.write_bytes(int8_bytes)
        int8_size_mb = len(int8_bytes) / 1024 / 1024
        print(f"  Saved: {TFLITE_INT8}  ({int8_size_mb:.2f} MB)  "
              f"[{time.time()-t0:.1f}s]")
        print("  Evaluating INT8 accuracy on test split...")
        int8_acc, _ = evaluate_tflite_accuracy(int8_bytes)
        print(f"  INT8 test accuracy: {int8_acc*100:.2f}%\n")

    finally:
        shutil.rmtree(tmp_savedmodel, ignore_errors=True)

    # ------------------------------------------------------------------
    # 5. Validate INT8 outputs against reference
    # ------------------------------------------------------------------
    print("Step 5: Validating INT8 model outputs against reference...")
    valid = validate_outputs(source, int8_bytes, calibration_images)
    print()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    acc_drop = (float32_acc - int8_acc) * 100
    size_reduction = float32_size_mb / int8_size_mb if int8_size_mb > 0 else 0
    print("=== Quantization Summary ===")
    print(f"  Float32 model size : {float32_size_mb:.2f} MB  →  {TFLITE_FLOAT32.name}")
    print(f"  INT8 model size    : {int8_size_mb:.2f} MB  →  {TFLITE_INT8.name}")
    print(f"  Size reduction     : {size_reduction:.1f}x")
    print(f"  Float32 accuracy   : {float32_acc*100:.2f}%")
    print(f"  INT8 accuracy      : {int8_acc*100:.2f}%  (drop: {acc_drop:.2f}pp)")
    print(f"  Output validation  : {'PASSED' if valid else 'FAILED'}")
    print(f"\n  Deploy {TFLITE_INT8.name} to the Raspberry Pi for Phase 4.\n")

    if int8_size_mb > 10:
        print(f"  WARNING: INT8 model ({int8_size_mb:.1f} MB) exceeds 10 MB target. "
              "Consider reducing IMAGE_SIZE or using a smaller base model.\n")


if __name__ == "__main__":
    quantize()
