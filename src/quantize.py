"""
quantize.py — INT8 post-training quantization and ONNX export.

Converts the trained PyTorch checkpoint to two ONNX variants:
  1. Float32 baseline  → models/plant_disease_float32.onnx
  2. INT8 quantized    → models/plant_disease.onnx  (final deployment model)

Conversion path: .pt checkpoint → ONNX float32 → ONNX INT8 (via onnxruntime.quantization).

INT8 quantization uses static quantization with a representative dataset of
200 training-set images to calibrate activation ranges.

Usage:
    conda run -n pydl python3 src/quantize.py

Outputs:
    models/plant_disease_float32.onnx  — unquantized baseline
    models/plant_disease.onnx          — INT8 quantized (deploy this)
    models/labels.txt                  — class names (already present)

Requirements:
    torch, torchvision, onnx, onnxruntime (same env as training)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINT_DIR: Path = MODELS_DIR / "checkpoints"
LABELS_FILE: Path = MODELS_DIR / "labels.txt"
DATA_TRAIN_DIR: Path = PROJECT_ROOT / "data" / "processed" / "train"
DATA_TEST_DIR: Path = PROJECT_ROOT / "data" / "processed" / "test"

ONNX_FLOAT32: Path = MODELS_DIR / "plant_disease_float32.onnx"
ONNX_INT8: Path = MODELS_DIR / "plant_disease.onnx"
ONNX_PREPROCESSED: Path = MODELS_DIR / "_preprocessed_for_quant.onnx"

NUM_CALIBRATION_SAMPLES: int = 200
IMAGE_SIZE: tuple[int, int] = (224, 224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names(labels_path: Path) -> list[str]:
    return [
        line.strip()
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_source_model() -> tuple[nn.Module, list[str]]:
    """Load the best available PyTorch checkpoint."""
    for ckpt_path in (
        CHECKPOINT_DIR / "best_stage2.pt",
        CHECKPOINT_DIR / "best_stage1.pt",
    ):
        if ckpt_path.exists():
            print(f"  Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
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
            return model, class_names

    raise FileNotFoundError(
        "No .pt checkpoint found. Run src/train.py first.\n"
        f"  Looked for: {CHECKPOINT_DIR / 'best_stage2.pt'}\n"
        f"             {CHECKPOINT_DIR / 'best_stage1.pt'}"
    )


def preprocess_image(path: Path) -> np.ndarray:
    """Load, resize, and normalise one image to NCHW float32."""
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.astype(np.float32)  # (3, H, W)


def collect_calibration_images(num_samples: int = NUM_CALIBRATION_SAMPLES) -> list[np.ndarray]:
    """
    Collect a balanced sample of training images for INT8 calibration.

    Returns list of float32 numpy arrays shaped (3, H, W) — one per image.
    """
    class_dirs = sorted(d for d in DATA_TRAIN_DIR.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class dirs found in {DATA_TRAIN_DIR}. Run data_prep.py.")

    per_class = max(1, num_samples // len(class_dirs))
    images: list[np.ndarray] = []

    for cls_dir in class_dirs:
        for path in sorted(cls_dir.glob("*.jpg"))[:per_class]:
            images.append(preprocess_image(path))

    print(
        f"  Calibration images: {len(images)} "
        f"(~{len(images) // len(class_dirs)} per class, {len(class_dirs)} classes)"
    )
    return images


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: nn.Module, output_path: Path) -> None:
    """Export PyTorch model to ONNX float32."""
    dummy = torch.randn(1, 3, *IMAGE_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        dynamo=False,        # use legacy TorchScript exporter, no onnxscript needed
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Exported ONNX: {output_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# INT8 static quantization
# ---------------------------------------------------------------------------

class _CalibrationReader:
    """CalibrationDataReader compatible with onnxruntime.quantization API."""

    def __init__(self, images: list[np.ndarray]):
        self._images = images
        self._index = 0

    def get_next(self):
        if self._index >= len(self._images):
            return None
        img = self._images[self._index][np.newaxis, ...]  # (1, 3, H, W)
        self._index += 1
        return {"input": img}

    def rewind(self):
        self._index = 0


def quantize_to_int8(
    float32_onnx_path: Path,
    int8_onnx_path: Path,
    calibration_images: list[np.ndarray],
) -> None:
    """Apply INT8 static quantization to the ONNX float32 model."""
    import onnx
    from onnxruntime.quantization import (
        quantize_static,
        QuantType,
        QuantFormat,
        quant_pre_process,
        CalibrationDataReader,
    )

    # Pre-process: add shape inference info required by the quantizer
    print("  Pre-processing ONNX graph for quantization...")
    quant_pre_process(str(float32_onnx_path), str(ONNX_PREPROCESSED), skip_optimization=False)

    # quant_pre_process can rename the input tensor (e.g. "input" → "input.1").
    # Read the actual name from the preprocessed model so calibration data is
    # delivered under the correct key — wrong key = zero calibration stats =
    # all activation ranges collapse = model always predicts the same class.
    preprocessed_model = onnx.load(str(ONNX_PREPROCESSED))
    actual_input_name = preprocessed_model.graph.input[0].name
    print(f"  Preprocessed model input name: '{actual_input_name}'")

    class _Reader(CalibrationDataReader):
        def __init__(self, images, inp_name):
            self._images = images
            self._inp_name = inp_name
            self._index = 0

        def get_next(self):
            if self._index >= len(self._images):
                return None
            img = self._images[self._index][np.newaxis, ...]
            self._index += 1
            return {self._inp_name: img}

    reader = _Reader(calibration_images, actual_input_name)

    print("  Running static INT8 quantization (this may take a minute)...")
    quantize_static(
        model_input=str(ONNX_PREPROCESSED),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    # Clean up temporary preprocessed model
    ONNX_PREPROCESSED.unlink(missing_ok=True)

    size_mb = int8_onnx_path.stat().st_size / 1024 / 1024
    print(f"  Saved INT8 ONNX: {int8_onnx_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_onnx_accuracy(
    onnx_path: Path,
    split: str = "test",
    max_images: int = 500,
) -> tuple[float, list[str]]:
    """
    Evaluate ONNX model top-1 accuracy on a data split.

    Class order is derived from the split directory (alphabetical), which
    matches ImageFolder's ordering used during training.

    Returns:
        (top-1 accuracy in [0, 1], alphabetically-sorted class names)
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    split_dir = PROJECT_ROOT / "data" / "processed" / split
    class_names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())

    per_class = max(1, max_images // len(class_names))
    correct = 0
    total = 0

    for class_idx, class_name in enumerate(class_names):
        cls_dir = split_dir / class_name
        for path in sorted(cls_dir.glob("*.jpg"))[:per_class]:
            img = preprocess_image(path)[np.newaxis, ...]  # (1, 3, H, W)
            outputs = session.run(None, {input_name: img})
            pred = int(np.argmax(outputs[0][0]))
            if pred == class_idx:
                correct += 1
            total += 1

    return (correct / total if total > 0 else 0.0), class_names


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def validate_outputs(
    source: nn.Module,
    onnx_path: Path,
    calibration_images: list[np.ndarray],
    num_checks: int = 10,
    tolerance: float = 0.10,
) -> bool:
    """
    Verify ONNX model outputs closely match PyTorch model outputs.

    Compares softmax probabilities (not raw logits) so the tolerance is
    meaningful — logits can differ by several units while probabilities stay
    within a few percent.

    Returns True if all top-1 predictions match within confidence tolerance.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(calibration_images), size=min(num_checks, len(calibration_images)), replace=False)

    mismatches = 0
    with torch.no_grad():
        for i in indices:
            img_np = calibration_images[i][np.newaxis, ...].astype(np.float32)
            img_pt = torch.from_numpy(img_np)

            # PyTorch reference — apply softmax to convert logits → probabilities
            ref_logits = source(img_pt).numpy()[0]
            ref_probs  = _softmax(ref_logits)
            ref_top1   = int(np.argmax(ref_probs))
            ref_conf   = float(ref_probs[ref_top1])

            # ONNX — also apply softmax (model outputs raw logits)
            ort_logits = session.run(None, {input_name: img_np})[0][0]
            ort_probs  = _softmax(ort_logits)
            ort_top1   = int(np.argmax(ort_probs))
            ort_conf   = float(ort_probs[ort_top1])

            match = ref_top1 == ort_top1
            diff = abs(ref_conf - ort_conf)
            status = "✓" if match and diff <= tolerance else "✗"
            print(
                f"  [{status}] sample {i:3d}: ref={ref_top1} ({ref_conf:.3f})  "
                f"onnx={ort_top1} ({ort_conf:.3f})  Δconf={diff:.4f}"
            )
            if not (match and diff <= tolerance):
                mismatches += 1

    passed = num_checks - mismatches
    print(f"\n  Validation: {passed}/{num_checks} checks passed (tolerance={tolerance:.2f})")
    return mismatches == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def quantize() -> None:
    """
    Full Phase 3 pipeline:
      1. Load trained PyTorch model and export to ONNX float32.
      2. Evaluate float32 ONNX accuracy on test split.
      3. Collect calibration images.
      4. Apply INT8 static quantization.
      5. Evaluate INT8 accuracy on test split.
      6. Validate INT8 outputs against reference.
      7. Print summary.
    """
    print("=== Phase 3: ONNX Quantization ===\n")

    class_names = load_class_names(LABELS_FILE)
    print(f"Classes ({len(class_names)}): {class_names}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model and export to ONNX float32
    # ------------------------------------------------------------------
    print("Step 1: Loading trained model and exporting to ONNX float32...")
    source, ckpt_classes = load_source_model()
    print(f"  Classes from checkpoint: {ckpt_classes}")
    t0 = time.time()
    export_onnx(source, ONNX_FLOAT32)
    float32_size_mb = ONNX_FLOAT32.stat().st_size / 1024 / 1024
    print(f"  Export took {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # 2. Float32 accuracy
    # ------------------------------------------------------------------
    print("Step 2: Evaluating float32 ONNX accuracy on test split...")
    float32_acc, eval_classes = evaluate_onnx_accuracy(ONNX_FLOAT32)
    print(f"  Float32 test accuracy: {float32_acc*100:.2f}%")
    print(f"  Class order used for eval: {eval_classes}\n")

    # ------------------------------------------------------------------
    # 3. Collect calibration images
    # ------------------------------------------------------------------
    print("Step 3: Collecting calibration images...")
    calibration_images = collect_calibration_images(NUM_CALIBRATION_SAMPLES)
    print()

    # ------------------------------------------------------------------
    # 4. INT8 quantization
    # ------------------------------------------------------------------
    print("Step 4: Applying INT8 static quantization...")
    t0 = time.time()
    quantize_to_int8(ONNX_FLOAT32, ONNX_INT8, calibration_images)
    int8_size_mb = ONNX_INT8.stat().st_size / 1024 / 1024
    print(f"  Quantization took {time.time()-t0:.1f}s\n")

    # ------------------------------------------------------------------
    # 5. INT8 accuracy
    # ------------------------------------------------------------------
    print("Step 5: Evaluating INT8 ONNX accuracy on test split...")
    int8_acc, _ = evaluate_onnx_accuracy(ONNX_INT8)
    print(f"  INT8 test accuracy: {int8_acc*100:.2f}%\n")

    # ------------------------------------------------------------------
    # 6. Output validation
    # ------------------------------------------------------------------
    print("Step 6: Validating INT8 model outputs against PyTorch reference...")
    valid = validate_outputs(source, ONNX_INT8, calibration_images)
    print()

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    acc_drop = (float32_acc - int8_acc) * 100
    size_reduction = float32_size_mb / int8_size_mb if int8_size_mb > 0 else 0
    print("=== Quantization Summary ===")
    print(f"  Float32 size  : {float32_size_mb:.2f} MB  →  {ONNX_FLOAT32.name}")
    print(f"  INT8 size     : {int8_size_mb:.2f} MB  →  {ONNX_INT8.name}")
    print(f"  Size reduction: {size_reduction:.1f}x")
    print(f"  Float32 acc   : {float32_acc*100:.2f}%")
    print(f"  INT8 acc      : {int8_acc*100:.2f}%  (drop: {acc_drop:.2f}pp)")
    print(f"  Validation    : {'PASSED' if valid else 'FAILED'}")
    print(f"\n  Deploy {ONNX_INT8.name} to the Raspberry Pi for Phase 4.\n")

    if int8_size_mb > 10:
        print(
            f"  WARNING: INT8 model ({int8_size_mb:.1f} MB) exceeds 10 MB target. "
            "Consider reducing IMAGE_SIZE or using a smaller backbone.\n"
        )


if __name__ == "__main__":
    quantize()
