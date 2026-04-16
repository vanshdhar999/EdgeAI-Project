"""
benchmark.py — TFLite inference latency benchmarking.

Measures per-frame inference time for the INT8 and float32 TFLite models.
Run on the GPU/dev machine first for a CPU baseline, then on the Pi for the
actual deployment target numbers.

Usage:
    # On dev machine (CPU baseline):
    python3 src/benchmark.py

    # On Raspberry Pi (deployment target):
    python3 src/benchmark.py

Output:
    Latency statistics (mean, median, p95, p99, min, max) for both models.
    Results printed to stdout.

Requirements:
    pip install tflite-runtime   (Pi)
    pip install tensorflow        (dev machine — tf.lite is included)
"""

import sys
import time
import platform
from pathlib import Path

import numpy as np

# Use tflite-runtime if available (lightweight, Pi-compatible),
# fall back to tf.lite on dev machine where full TF is installed.
try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    _runtime = "tflite-runtime"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    _runtime = "tf.lite"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"

TFLITE_INT8: Path = MODELS_DIR / "plant_disease.tflite"
TFLITE_FLOAT32: Path = MODELS_DIR / "plant_disease_float32.tflite"

IMAGE_SIZE: tuple[int, int] = (224, 224)

# Number of warmup runs (not included in timing)
WARMUP_RUNS: int = 5

# Number of timed inference runs
BENCHMARK_RUNS: int = 100


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(
    model_path: Path,
    num_runs: int = BENCHMARK_RUNS,
    num_warmup: int = WARMUP_RUNS,
) -> dict[str, float]:
    """
    Load a TFLite model and measure per-inference latency over multiple runs.

    Args:
        model_path: Path to the .tflite file.
        num_runs:   Number of timed inference calls.
        num_warmup: Number of warmup calls (excluded from timing).

    Returns:
        Dict with latency statistics in milliseconds:
        mean, median, p95, p99, min, max, std.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"TFLite model not found: {model_path}\n"
            "Run src/quantize.py first."
        )

    interp = Interpreter(model_path=str(model_path))
    interp.allocate_tensors()

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    # Dummy input: random float32 image in [0, 1] — shape matches model input
    inp_shape = inp_details[0]["shape"]   # e.g. [1, 224, 224, 3]
    dummy = np.random.default_rng(seed=0).random(inp_shape).astype(np.float32)

    # Warmup (fills caches, JIT-compiles any lazy ops)
    for _ in range(num_warmup):
        interp.set_tensor(inp_details[0]["index"], dummy)
        interp.invoke()

    # Timed runs
    latencies_ms: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp_details[0]["index"], dummy)
        interp.invoke()
        _ = interp.get_tensor(out_details[0]["index"])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "std":    float(np.std(arr)),
    }


def print_stats(label: str, stats: dict[str, float], model_path: Path) -> None:
    """Print formatted latency statistics for one model."""
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"\n  {label}  ({size_mb:.2f} MB)")
    print(f"  {'─' * 40}")
    print(f"  Mean   : {stats['mean']:7.2f} ms")
    print(f"  Median : {stats['median']:7.2f} ms")
    print(f"  p95    : {stats['p95']:7.2f} ms")
    print(f"  p99    : {stats['p99']:7.2f} ms")
    print(f"  Min    : {stats['min']:7.2f} ms")
    print(f"  Max    : {stats['max']:7.2f} ms")
    print(f"  Std    : {stats['std']:7.2f} ms")

    target_ms = 1500  # Phase 3 success criterion: <1.5s on Pi
    if stats["mean"] < target_ms:
        print(f"  ✓ Mean latency {stats['mean']:.0f} ms is within {target_ms} ms target")
    else:
        print(f"  ✗ Mean latency {stats['mean']:.0f} ms exceeds {target_ms} ms target")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark() -> None:
    """
    Benchmark available TFLite models and print latency statistics.

    Run this on the dev machine first (CPU baseline), then copy the .tflite
    files to the Pi and run again for the actual deployment target numbers.
    """
    print("=== TFLite Latency Benchmark ===\n")
    print(f"  Runtime   : {_runtime}")
    print(f"  Platform  : {platform.machine()} / {platform.system()} {platform.release()}")
    print(f"  CPU count : {__import__('os').cpu_count()}")
    print(f"  Warmup    : {WARMUP_RUNS} runs")
    print(f"  Benchmark : {BENCHMARK_RUNS} runs")
    print(f"  Input     : {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}×3 float32\n")

    results: dict[str, dict] = {}

    if TFLITE_INT8.exists():
        print(f"Benchmarking INT8 model ({TFLITE_INT8.name})...")
        results["INT8"] = benchmark_model(TFLITE_INT8)
        print_stats("INT8  (plant_disease.tflite)", results["INT8"], TFLITE_INT8)
    else:
        print(f"  [SKIP] INT8 model not found: {TFLITE_INT8}")

    if TFLITE_FLOAT32.exists():
        print(f"\nBenchmarking float32 model ({TFLITE_FLOAT32.name})...")
        results["float32"] = benchmark_model(TFLITE_FLOAT32)
        print_stats("Float32 baseline", results["float32"], TFLITE_FLOAT32)
    else:
        print(f"  [SKIP] Float32 model not found: {TFLITE_FLOAT32}")

    if "INT8" in results and "float32" in results:
        speedup = results["float32"]["mean"] / results["INT8"]["mean"]
        print(f"\n  INT8 speedup vs float32: {speedup:.2f}x  "
              f"(mean {results['float32']['mean']:.1f} ms → {results['INT8']['mean']:.1f} ms)\n")

    print("─" * 50)
    print("Run this script on the Raspberry Pi for deployment target numbers.")
    print("Target: mean latency < 1500 ms on Pi 5 (expected ~200–400 ms).\n")


if __name__ == "__main__":
    benchmark()
