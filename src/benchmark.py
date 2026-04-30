"""
benchmark.py — ONNX Runtime inference latency benchmarking.

Measures per-frame inference time for the INT8 and float32 ONNX models.
Run on the dev machine first for a CPU baseline, then on the Pi for
the actual deployment target numbers.

Usage:
    # On dev machine (CPU baseline):
    conda run -n pydl python3 src/benchmark.py

    # On Raspberry Pi (deployment target):
    python3 src/benchmark.py

Output:
    Latency statistics (mean, median, p95, p99, min, max) for both models.

Requirements:
    onnxruntime (dev machine and Pi)
"""

import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = PROJECT_ROOT / "models"

ONNX_INT8: Path = MODELS_DIR / "plant_disease.onnx"
ONNX_FLOAT32: Path = MODELS_DIR / "plant_disease_float32.onnx"

IMAGE_SIZE: tuple[int, int] = (224, 224)

WARMUP_RUNS: int = 5
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
    Load an ONNX model and measure per-inference latency.

    Returns:
        Dict with latency statistics in milliseconds:
        mean, median, p95, p99, min, max, std.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}\n"
            "Run src/quantize.py first."
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape  # [batch, 3, H, W]

    # Replace dynamic batch dim with 1
    shape = [1 if isinstance(d, str) else d for d in input_shape]
    dummy = np.random.default_rng(seed=0).random(shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    # Timed runs
    latencies_ms: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
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

    target_ms = 1500
    if stats["mean"] < target_ms:
        print(f"  ✓ Mean latency {stats['mean']:.0f} ms within {target_ms} ms target")
    else:
        print(f"  ✗ Mean latency {stats['mean']:.0f} ms exceeds {target_ms} ms target")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def benchmark() -> None:
    """Benchmark available ONNX models and print latency statistics."""
    ort_version = ort.__version__
    print("=== ONNX Runtime Latency Benchmark ===\n")
    print(f"  ONNX Runtime : {ort_version}")
    print(f"  Platform     : {platform.machine()} / {platform.system()} {platform.release()}")
    print(f"  CPU count    : {__import__('os').cpu_count()}")
    print(f"  Providers    : {ort.get_available_providers()}")
    print(f"  Warmup       : {WARMUP_RUNS} runs")
    print(f"  Benchmark    : {BENCHMARK_RUNS} runs")
    print(f"  Input        : 1×3×{IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} float32 (NCHW)\n")

    results: dict[str, dict] = {}

    if ONNX_INT8.exists():
        print(f"Benchmarking INT8 model ({ONNX_INT8.name})...")
        results["INT8"] = benchmark_model(ONNX_INT8)
        print_stats("INT8  (plant_disease.onnx)", results["INT8"], ONNX_INT8)
    else:
        print(f"  [SKIP] INT8 model not found: {ONNX_INT8}")

    if ONNX_FLOAT32.exists():
        print(f"\nBenchmarking float32 model ({ONNX_FLOAT32.name})...")
        results["float32"] = benchmark_model(ONNX_FLOAT32)
        print_stats("Float32 baseline", results["float32"], ONNX_FLOAT32)
    else:
        print(f"  [SKIP] Float32 model not found: {ONNX_FLOAT32}")

    if "INT8" in results and "float32" in results:
        speedup = results["float32"]["mean"] / results["INT8"]["mean"]
        print(
            f"\n  INT8 speedup vs float32: {speedup:.2f}x  "
            f"(mean {results['float32']['mean']:.1f} ms → {results['INT8']['mean']:.1f} ms)\n"
        )

    print("─" * 50)
    print("Run this script on the Raspberry Pi for deployment target numbers.")
    print("Target: mean latency < 1500 ms on Pi 5 (expected ~100–400 ms with ONNX Runtime).\n")


if __name__ == "__main__":
    benchmark()
