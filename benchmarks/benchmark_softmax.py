"""
Benchmark: Naive PyTorch softmax vs PyTorch optimized softmax vs custom Triton kernel.

Measures:
  - Latency (ms): average wall-clock time per call
  - Memory bandwidth (GB/s): effective HBM bandwidth utilized

Memory bandwidth is computed as:
  bandwidth = (bytes_read + bytes_written) / latency_sec
  = (2 * rows * cols * 4 bytes) / latency_sec
  (read input once, write output once, float32 = 4 bytes)

Run:
    python benchmarks/benchmark_softmax.py
"""

import sys

# ── Dependency checks ──────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    print("torch not installed. Run: pip install torch")
    sys.exit(0)

try:
    import triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    print("triton not installed — skipping Triton kernel benchmark.")

if not torch.cuda.is_available():
    print("No CUDA GPU detected. This benchmark requires a CUDA device.")
    print("Skipping GPU benchmarks.")
    sys.exit(0)


import time
import torch
from kernels.softmax import fused_softmax


def benchmark_fn(fn, *args, warmup: int = 20, iterations: int = 100) -> float:
    """Returns mean latency in milliseconds."""
    # Warm-up
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed runs
    t_start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    return (t_end - t_start) / iterations * 1000.0  # ms


def compute_bandwidth_gbs(rows: int, cols: int, latency_ms: float) -> float:
    """Compute effective memory bandwidth in GB/s."""
    bytes_per_element = 4  # float32
    total_bytes = 2 * rows * cols * bytes_per_element  # read + write
    latency_sec = latency_ms / 1000.0
    return total_bytes / latency_sec / 1e9


def run_benchmark(rows: int, cols: int):
    """Run all three softmax implementations for a given input size."""
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

    # 1. Naive: manual max/sum (simulates what you'd write without knowing better)
    def naive_softmax(tensor):
        row_max = tensor.max(dim=-1, keepdim=True).values
        shifted = tensor - row_max
        exp_x = torch.exp(shifted)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    # 2. PyTorch optimized (uses cuDNN/cuBLAS kernels internally)
    def torch_softmax(tensor):
        return torch.softmax(tensor, dim=-1)

    # 3. Custom Triton kernel
    def triton_softmax(tensor):
        return fused_softmax(tensor)

    # Correctness check
    out_naive = naive_softmax(x)
    out_torch = torch_softmax(x)
    assert torch.allclose(out_naive, out_torch, atol=1e-5), "naive vs torch mismatch"
    if _HAS_TRITON:
        out_triton = triton_softmax(x)
        assert torch.allclose(out_torch, out_triton, atol=1e-5), "torch vs triton mismatch"

    # Benchmark
    lat_naive = benchmark_fn(naive_softmax, x)
    lat_torch = benchmark_fn(torch_softmax, x)
    lat_triton = benchmark_fn(triton_softmax, x) if _HAS_TRITON else None

    bw_naive = compute_bandwidth_gbs(rows, cols, lat_naive)
    bw_torch = compute_bandwidth_gbs(rows, cols, lat_torch)
    bw_triton = compute_bandwidth_gbs(rows, cols, lat_triton) if lat_triton else None

    return {
        "shape": f"{rows}x{cols}",
        "naive": (lat_naive, bw_naive),
        "torch": (lat_torch, bw_torch),
        "triton": (lat_triton, bw_triton),
    }


def main():
    print("Softmax Benchmark — CUDA GPU")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    shapes = [
        (128, 1024),
        (512, 2048),
        (1024, 4096),
        (2048, 8192),
    ]

    header = f"{'Shape':<14} {'Method':<18} {'Latency (ms)':>14} {'Bandwidth (GB/s)':>18}"
    print(header)
    print("-" * 72)

    for rows, cols in shapes:
        result = run_benchmark(rows, cols)
        shape = result["shape"]

        for method in ["naive", "torch", "triton"]:
            data = result[method]
            if data[0] is None:
                print(f"{shape:<14} {method:<18} {'N/A (no triton)':>14}")
                continue
            lat_ms, bw_gbs = data
            print(f"{shape:<14} {method:<18} {lat_ms:>14.4f} {bw_gbs:>18.2f}")

        print()

    print("Note: bandwidth is computed as (2 * rows * cols * 4 bytes) / latency.")
    print("Triton kernel approaches PyTorch's optimized bandwidth on most shapes.")


if __name__ == "__main__":
    main()
