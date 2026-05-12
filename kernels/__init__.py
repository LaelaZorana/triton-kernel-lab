"""
triton-kernel-lab: Custom GPU kernels implemented in OpenAI Triton.

Kernels:
  - softmax: Fused numerically-stable softmax
  - elementwise: ReLU and GELU approximation
  - matrix_ops: Tiled matrix multiplication

Each module provides a Python dispatch function alongside the @triton.jit kernel.
Requires CUDA GPU + triton>=2.0 + torch>=2.0 to run.
"""
