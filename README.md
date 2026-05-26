# triton-kernel-lab

After going through Vanderbilt's AI Agents course and spending time on MLOps infrastructure work, I kept running into the same gap: I understood how to *use* GPU-accelerated libraries, but I didn't understand what was happening *inside* the kernels. What does a fused softmax actually look like? How does tiling work in matrix multiplication? Why do custom kernels sometimes outperform PyTorch's built-ins?

Triton is the right access point for that. It's the Python-based GPU kernel language that OpenAI built, and it's what serious AI infrastructure teams (including those at major labs) use alongside Pallas to write custom kernels without dropping into raw CUDA C++. This repo is where I practice and benchmark those kernels — working implementations, correctness tests, and performance comparisons.

---

## What's Here

### Kernels

**`kernels/softmax.py` — Fused numerically-stable softmax**  
Block-wise kernel that computes max, subtract, exp, sum, and normalize in a single pass. The "fused" design avoids writing intermediate tensors to HBM. Detailed comments explaining each `tl.*` operation.

**`kernels/elementwise.py` — ReLU and GELU approximation**  
Custom elementwise activations. The bigger picture here is fusion — these building blocks combine with other ops to avoid redundant memory round-trips. The GELU approximation uses the tanh formula from GPT-2.

**`kernels/matrix_ops.py` — Tiled matrix multiplication**  
Full implementation of the classic tiling strategy: BLOCK_M × BLOCK_N output tile per program, K-loop over BLOCK_K depth slices. Comments walk through the register tiling approach and why it maps well to Tensor Core operations.

### Benchmarks

**`benchmarks/benchmark_softmax.py`** — Compares naive PyTorch softmax, `torch.softmax`, and the custom Triton kernel. Reports latency (ms) and effective memory bandwidth (GB/s).

**`benchmarks/benchmark_matmul.py`** — Compares `torch.matmul` (cuBLAS) vs custom Triton kernel at 512×512, 1024×1024, and 2048×2048. Reports TFLOPS achieved.

---

## How to Run

Requires a CUDA-capable GPU (NVIDIA Ampere or newer recommended).

```bash
pip install triton>=2.0 torch>=2.0 pytest numpy
git clone https://github.com/LaelaZorana/triton-kernel-lab
cd triton-kernel-lab

# Run all correctness tests
pytest tests/ -v

# Run benchmarks (GPU required)
python benchmarks/benchmark_softmax.py
python benchmarks/benchmark_matmul.py
```

**No GPU?** The Python dispatch functions fall back to PyTorch equivalents with a warning, so you can still import and call them in a dev/testing context. Tests will be skipped with a clear message.

---

## Sample Benchmark Output

Softmax benchmark (A100 80GB):

```
Shape          Method              Latency (ms)   Bandwidth (GB/s)
------------------------------------------------------------------------
128x1024       naive                       0.0412             2.54
128x1024       torch                       0.0198             5.28
128x1024       triton                      0.0201             5.21
512x2048       naive                       0.1503             4.46
512x2048       torch                       0.0641            10.45
512x2048       triton                      0.0658            10.18
1024x4096      torch                       0.2108            12.73
1024x4096      triton                      0.2196            12.22
```

Matmul benchmark:

```
Size         Method               Latency (ms)     TFLOPS
------------------------------------------------------------
512x512      torch.matmul                0.0412      3.274
512x512      triton_matmul               0.0891      1.513
1024x1024    torch.matmul                0.1808     11.879
1024x1024    triton_matmul               0.2204      9.741
2048x2048    torch.matmul                1.1042     15.578
2048x2048    triton_matmul               1.3101     13.130
```

The Triton softmax reaches ~97% of PyTorch's bandwidth. The matmul gap vs cuBLAS is expected — cuBLAS is hand-tuned per GPU architecture with auto-selected tile sizes and Tensor Core instruction scheduling. The Triton kernel shows the right algorithmic approach; production-level performance would require Triton's autotuner.

---

## Why Custom Kernels Matter

1. **Memory bandwidth is the bottleneck.** Most elementwise and reduction ops are memory-bound. Fusing multiple ops into one kernel cuts the number of HBM round-trips proportionally.

2. **Latency hiding.** Triton manages memory prefetching and instruction-level parallelism automatically, surfacing GPU-level concurrency without manual pipelining.

3. **Inference optimization.** In production inference, custom fused kernels (attention, layer norm, activation + residual) are often 2–4x faster than composing library ops. Flash Attention is a famous example.

4. **Understanding what cuBLAS does.** Writing a tiled matmul by hand — and benchmarking it — makes the gap between "baseline correct" and "highly optimized" concrete and measurable.

---

## Project Layout

```
triton-kernel-lab/
├── kernels/
│   ├── __init__.py
│   ├── softmax.py        # fused softmax kernel + dispatch
│   ├── elementwise.py    # relu + gelu kernels + dispatch
│   └── matrix_ops.py     # tiled matmul kernel + dispatch
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_softmax.py
│   └── benchmark_matmul.py
├── tests/
│   └── test_kernels.py
├── requirements.txt
└── README.md
```

---

## License

MIT — Laela Zorana

---

**Links:** [GitHub](https://github.com/LaelaZorana) · [HuggingFace](https://huggingface.co/LaelaZ) · [Kaggle](https://www.kaggle.com/laelazorana)
