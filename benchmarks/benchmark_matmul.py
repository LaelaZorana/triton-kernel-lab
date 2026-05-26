"""
Benchmark: PyTorch matmul vs custom Triton tiled matmul kernel.

Measures TFLOPS achieved at square matrix sizes: 512, 1024, 2048.

TFLOPS formula for square matmul [N, N] @ [N, N]:
  FLOPs = 2 * N^3  (N^3 multiply-accumulate = 2*N^3 FLOPs)
  TFLOPS = FLOPs / latency_sec / 1e12

Run:
    python benchmarks/benchmark_matmul.py
"""

import sys

try:
    import torch
except ImportError:
    print("torch not installed.")
    sys.exit(0)

try:
    import triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    print("triton not installed — skipping Triton matmul benchmark.")

if not torch.cuda.is_available():
    print("No CUDA device detected. Skipping GPU benchmark.")
    sys.exit(0)

import time
from kernels.matrix_ops import triton_matmul


def benchmark_fn(fn, *args, warmup: int = 10, iterations: int = 50) -> float:
    """Returns mean latency in milliseconds."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iterations * 1000.0


def tflops(n: int, latency_ms: float) -> float:
    """Compute TFLOPS for an N×N square matmul."""
    flops = 2 * (n ** 3)
    latency_sec = latency_ms / 1000.0
    return flops / latency_sec / 1e12


def main():
    print("Matrix Multiplication Benchmark — CUDA GPU")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    sizes = [512, 1024, 2048]

    header = f"{'Size':<12} {'Method':<20} {'Latency (ms)':>14} {'TFLOPS':>10}"
    print(header)
    print("-" * 60)

    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)

        # PyTorch matmul (uses cuBLAS)
        lat_torch = benchmark_fn(torch.matmul, a, b)
        tf_torch = tflops(n, lat_torch)
        print(f"{n}x{n}      {'torch.matmul':<20} {lat_torch:>14.4f} {tf_torch:>10.3f}")

        # Custom Triton kernel
        if _HAS_TRITON:
            # Verify correctness
            c_torch = torch.matmul(a, b)
            c_triton = triton_matmul(a, b)
            if not torch.allclose(c_torch, c_triton, atol=1e-3):
                print(f"  WARNING: Triton result differs from torch for {n}x{n} (max diff: {(c_torch - c_triton).abs().max():.4f})")

            lat_triton = benchmark_fn(triton_matmul, a, b)
            tf_triton = tflops(n, lat_triton)
            print(f"{n}x{n}      {'triton_matmul':<20} {lat_triton:>14.4f} {tf_triton:>10.3f}")
        else:
            print(f"{n}x{n}      {'triton_matmul':<20} {'N/A':>14}")

        print()

    print("Note: torch.matmul uses cuBLAS which is heavily hand-tuned.")
    print("The Triton kernel demonstrates the approach; production tuning")
    print("would require autotuning BLOCK_M, BLOCK_N, BLOCK_K per GPU arch.")


if __name__ == "__main__":
    main()
