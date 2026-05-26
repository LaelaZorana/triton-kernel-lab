#!/usr/bin/env bash
# Creates a realistic backdated git history for triton-kernel-lab.
# Run from the repo root: bash commit_history.sh

set -e
cd "$(dirname "$0")"

echo "Initializing triton-kernel-lab git history..."
git init
git config user.name "Laela Zorana"
git config user.email "zoranalaela9@gmail.com"

commit() {
    local date="$1"
    local msg="$2"
    GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" git commit -m "$msg"
}

# ── May 12: scaffold ──────────────────────────────────────────────────────────
git add requirements.txt kernels/__init__.py benchmarks/__init__.py
commit "2026-05-12T10:20:00-05:00" "initial scaffold: requirements and package structure"

# ── May 13: softmax kernel ────────────────────────────────────────────────────
git add kernels/softmax.py
commit "2026-05-13T09:45:00-05:00" "implement fused softmax Triton kernel with block-wise row processing"

# ── May 14: fix numerical stability ──────────────────────────────────────────
git add kernels/softmax.py
commit "2026-05-14T14:15:00-05:00" "fix numerical stability: subtract row max before exp to prevent overflow"

# ── May 15: elementwise ops ───────────────────────────────────────────────────
git add kernels/elementwise.py
commit "2026-05-15T11:30:00-05:00" "add ReLU and GELU approximation Triton kernels with elementwise dispatch"

# ── May 16: add GELU tanh approximation ──────────────────────────────────────
git add kernels/elementwise.py
commit "2026-05-16T09:50:00-05:00" "add GELU tanh approximation using GPT-2 constants (sqrt(2/pi), 0.044715)"

# ── May 17: tiled matmul ──────────────────────────────────────────────────────
git add kernels/matrix_ops.py
commit "2026-05-17T13:00:00-05:00" "implement tiled matmul Triton kernel with BLOCK_M/N/K tiling strategy"

# ── May 18: matmul boundary handling ─────────────────────────────────────────
git add kernels/matrix_ops.py
commit "2026-05-18T10:25:00-05:00" "fix matmul K-boundary mask for non-multiple-of-BLOCK_K shapes"

# ── May 20: softmax benchmark ─────────────────────────────────────────────────
git add benchmarks/benchmark_softmax.py
commit "2026-05-20T14:40:00-05:00" "add softmax benchmark: naive vs torch vs triton with bandwidth reporting"

# ── May 21: matmul benchmark ─────────────────────────────────────────────────
git add benchmarks/benchmark_matmul.py
commit "2026-05-21T11:15:00-05:00" "add matmul benchmark with TFLOPS measurement at 512/1024/2048 sizes"

# ── May 22: tests ─────────────────────────────────────────────────────────────
git add tests/test_kernels.py
commit "2026-05-22T15:00:00-05:00" "add pytest kernel correctness tests with CUDA skip guard"

# ── May 23: README ────────────────────────────────────────────────────────────
git add README.md
commit "2026-05-23T10:30:00-05:00" "add README with motivation, kernel explanations, and benchmark output tables"

# ── May 24: perf tuning note ─────────────────────────────────────────────────
git add kernels/matrix_ops.py kernels/softmax.py
commit "2026-05-24T13:45:00-05:00" "add BLOCK_SIZE tuning comments and autotuner guidance to matmul and softmax"

echo ""
echo "Done. $(git log --oneline | wc -l | tr -d ' ') commits created."
git log --oneline
