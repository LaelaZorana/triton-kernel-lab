# triton-kernel-lab

After going through Vanderbilt's AI Agents course and spending time on MLOps infrastructure work, I kept running into the same gap: I understood how to *use* GPU-accelerated libraries, but I didn't understand what was happening *inside* the kernels. What does a fused softmax actually look like? How does tiling work in matrix multiplication? Why do custom kernels sometimes outperform PyTorch's built-ins?

Triton is the right access point for that, because it's the Python-based GPU kernel language that OpenAI built, and it's what serious AI infrastructure teams (including those at major labs) use alongside Pallas to write custom kernels without dropping into raw CUDA C++. This repo is where I practice and benchmark those kernels, so it holds working implementations, correctness tests, and performance comparisons.

This project was built in May 2026. In August 2026 I squashed the git history to a single commit during an account wide cleanup, so the commit date is newer than the work.

## What's Here

### Kernels

**`kernels/softmax.py`: Fused numerically-stable softmax**  
Block-wise kernel that computes max, subtract, exp, sum, and normalize in a single pass, and the "fused" design avoids writing intermediate tensors to HBM. Detailed comments explain each `tl.*` operation.

**`kernels/elementwise.py`: ReLU and GELU approximation**  
Custom elementwise activations, where the bigger picture is fusion, because these building blocks combine with other ops to avoid redundant memory round-trips. The GELU approximation uses the tanh formula from GPT-2.

**`kernels/matrix_ops.py`: Tiled matrix multiplication**  
Full implementation of the classic tiling strategy: BLOCK_M × BLOCK_N output tile per program, K-loop over BLOCK_K depth slices. Comments walk through the register tiling approach and why it maps well to Tensor Core operations.

### Benchmarks

**`benchmarks/benchmark_softmax.py`** compares naive PyTorch softmax, `torch.softmax`, and the custom Triton kernel, then reports latency (ms) and effective memory bandwidth (GB/s).

**`benchmarks/benchmark_matmul.py`** compares `torch.matmul` (cuBLAS) vs custom Triton kernel at 512×512, 1024×1024, and 2048×2048, then reports TFLOPS achieved.

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

## Benchmarks

The benchmark scripts in `benchmarks/` print latency, effective bandwidth for softmax, and TFLOPS for matmul across a sweep of shapes. I have not published numbers here yet because I want them measured on the hardware the table names, and the repo currently runs on machines without a data center GPU. Run them yourself with

```
python benchmarks/benchmark_softmax.py
python benchmarks/benchmark_matmul.py
```

What to expect from the shapes involved. The fused softmax should sit near PyTorch's bandwidth, since both are memory bound and the fusion removes the extra pass over the rows. The tiled matmul will trail cuBLAS, because cuBLAS is hand tuned per architecture with Tensor Core scheduling, and closing that gap needs Triton's autotuner rather than a fixed tile size.

## Why Custom Kernels Matter

1. **Memory bandwidth is the bottleneck.** Most elementwise and reduction ops are memory-bound. Fusing multiple ops into one kernel cuts the number of HBM round-trips proportionally.

2. **Latency hiding.** Triton manages memory prefetching and instruction-level parallelism automatically, surfacing GPU-level concurrency without manual pipelining.

3. **Inference optimization.** In production inference, custom fused kernels (attention, layer norm, activation + residual) are often 2 to 4x faster than composing library ops. Flash Attention is a famous example.

4. **Understanding what cuBLAS does.** Writing a tiled matmul by hand, then benchmarking it, makes the gap between "baseline correct" and "highly optimized" concrete and measurable.

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

## License

MIT, Laela Zorana

**Links:** [GitHub](https://github.com/LaelaZorana) · [HuggingFace](https://huggingface.co/LaelaZ) · [Kaggle](https://www.kaggle.com/laelazorana)
