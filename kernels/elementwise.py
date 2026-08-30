"""
Custom elementwise operation kernels: ReLU and GELU approximation.

Why write custom elementwise kernels?
--------------------------------------
PyTorch's built-in activations are already fast, but custom Triton kernels shine
when you want to FUSE multiple operations. For example, a dropout + ReLU + bias_add
fused kernel avoids three separate memory round-trips. Each unfused kernel reads
the full tensor from HBM and writes it back; fusing collapses N passes into one.

Memory bandwidth is usually the bottleneck for elementwise ops on modern GPUs
(they are "memory-bound"), not compute. Fusing 3 ops gives ~3x bandwidth savings.

This module shows the building blocks. In practice you'd combine these with
surrounding ops (e.g., linear + ReLU in one kernel).

Kernels:
  - relu_kernel: max(x, 0) elementwise
  - gelu_approx_kernel: GELU via tanh approximation (used in GPT-2, BERT)
"""

try:
    import triton
    import triton.language as tl
    import torch
    import math
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ── ReLU Kernel ────────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.jit
    def relu_kernel(
        x_ptr,          # input tensor pointer
        output_ptr,     # output tensor pointer
        n_elements,     # total number of elements
        BLOCK_SIZE: tl.constexpr,  # elements per program instance
    ):
        """
        Elementwise ReLU: output[i] = max(input[i], 0).

        Each Triton program handles a contiguous BLOCK_SIZE-element chunk.
        tl.program_id(0) gives the block index; we compute the absolute offsets
        and apply a mask to handle the last partial block.
        """
        # Block index → offset range for this program
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Mask: only process valid elements (last block may be smaller)
        mask = offsets < n_elements

        # Load input, apply ReLU, store
        x = tl.load(x_ptr + offsets, mask=mask)
        output = tl.maximum(x, 0.0)  # tl.maximum is elementwise, 0.0 broadcasts
        tl.store(output_ptr + offsets, output, mask=mask)


    @triton.jit
    def gelu_approx_kernel(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        This is the "fast GELU" used in GPT-2 and original BERT. The exact GELU
        uses the error function (erf), which is slower to compute.

        The tanh approximation is accurate to ~0.01% for values in [-3, 3] and
        is what most transformer implementations use in practice.

        Constants:
          sqrt(2/pi) ≈ 0.7978845608
          coefficient ≈ 0.044715
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)

        # GELU approximation constants
        SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2.0 / pi)
        COEFF = 0.044715

        # inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        cube = x * x * x
        inner = SQRT_2_OVER_PI * (x + COEFF * cube)

        # tanh is built into Triton's math library
        output = x * 0.5 * (1.0 + tl.libdevice.tanh(inner))

        tl.store(output_ptr + offsets, output, mask=mask)


# ── Python dispatch functions ──────────────────────────────────────────────────

def triton_relu(x: "torch.Tensor") -> "torch.Tensor":
    """
    ReLU activation via custom Triton kernel.

    Args:
        x: Float32 CUDA tensor of any shape.

    Returns:
        Tensor of same shape with ReLU applied.
    """
    if not _TRITON_AVAILABLE:
        raise ImportError("triton is not installed. pip install triton")
    if not x.is_cuda:
        import warnings
        warnings.warn("triton_relu requires CUDA tensor. Falling back to torch.relu.", stacklevel=2)
        return torch.relu(x)

    # Flatten to 1D for the kernel, reshape on return
    original_shape = x.shape
    x_flat = x.contiguous().view(-1)
    n = x_flat.numel()

    output = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024  # tunable; 1024 is a good default for elementwise ops
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    relu_kernel[grid](x_flat, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output.view(original_shape)


def triton_gelu(x: "torch.Tensor") -> "torch.Tensor":
    """
    GELU activation (tanh approximation) via custom Triton kernel.

    Args:
        x: Float32 CUDA tensor of any shape.

    Returns:
        Tensor of same shape with GELU applied.
    """
    if not _TRITON_AVAILABLE:
        raise ImportError("triton is not installed. pip install triton")
    if not x.is_cuda:
        import warnings
        warnings.warn("triton_gelu requires CUDA tensor. Falling back to torch.nn.functional.gelu.", stacklevel=2)
        import torch.nn.functional as F
        return F.gelu(x, approximate="tanh")

    original_shape = x.shape
    x_flat = x.contiguous().view(-1)
    n = x_flat.numel()

    output = torch.empty_like(x_flat)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    gelu_approx_kernel[grid](x_flat, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output.view(original_shape)
