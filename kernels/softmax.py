"""
Fused softmax kernel in OpenAI Triton.

Standard softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

The "fused" part means the entire computation (max reduction, subtract, exp,
sum reduction, divide) happens in a single kernel launch, no intermediate
device memory is written between steps. This saves bandwidth and reduces
kernel launch overhead compared to composing separate CUDA kernels.

Numerical stability: we subtract the row maximum before exponentiation.
Without this, large logits overflow float32 (exp(88) ≈ 1.65e38, near the limit).

Requires: CUDA GPU, triton>=2.0, torch>=2.0
"""

try:
    import triton
    import triton.language as tl
    import torch
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ── Triton kernel ──────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:

    @triton.jit
    def softmax_kernel(
        output_ptr,   # pointer to output tensor [rows, cols]
        input_ptr,    # pointer to input tensor  [rows, cols]
        input_row_stride,   # stride between rows in the input (== cols for contiguous)
        output_row_stride,  # stride between rows in the output
        n_cols,             # number of columns (elements per row)
        BLOCK_SIZE: tl.constexpr,  # power-of-2 tile width, chosen at dispatch time
    ):
        """
        Each Triton program instance handles one row of the input matrix.

        tl.program_id(0) gives us our row index.  We load the entire row into
        SRAM (shared/register memory), compute the block-level max and sum for
        numerical stability, then store the result.
        """
        # ── Step 1: identify our row ───────────────────────────────────────────
        row_idx = tl.program_id(axis=0)

        # Compute the starting memory offset for this row in the input and output
        row_start_ptr = input_ptr + row_idx * input_row_stride
        out_start_ptr = output_ptr + row_idx * output_row_stride

        # ── Step 2: build a column offset vector [0, 1, 2, ..., BLOCK_SIZE-1] ─
        col_offsets = tl.arange(0, BLOCK_SIZE)

        # Mask out columns beyond the actual row length (padding to BLOCK_SIZE)
        mask = col_offsets < n_cols

        # ── Step 3: load the row: masked load fills out-of-bounds with -inf ──
        # Using -inf as the default ensures masked values don't affect the max.
        row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))

        # ── Step 4: subtract max for numerical stability ───────────────────────
        # tl.max returns a scalar (reduction over the block dimension)
        row_max = tl.max(row, axis=0)
        row = row - row_max  # broadcast scalar subtraction across the block

        # ── Step 5: compute exp element-wise ──────────────────────────────────
        numerator = tl.exp(row)

        # ── Step 6: sum across the row ─────────────────────────────────────────
        denominator = tl.sum(numerator, axis=0)

        # ── Step 7: normalize ─────────────────────────────────────────────────
        softmax_output = numerator / denominator

        # ── Step 8: store result, masking out padding positions ───────────────
        tl.store(out_start_ptr + col_offsets, softmax_output, mask=mask)


# ── Python dispatch ────────────────────────────────────────────────────────────

def fused_softmax(x: "torch.Tensor") -> "torch.Tensor":
    """
    Compute softmax along the last dimension using the custom Triton kernel.

    For a 2D input of shape [rows, cols], each row is processed independently
    by one Triton program instance.

    Args:
        x: Float32 CUDA tensor of shape [rows, cols]. Must be contiguous.
           For higher-rank tensors, the last two dims are used after a reshape.

    Returns:
        Tensor of the same shape and dtype as x with softmax applied.

    Note:
        Requires a CUDA GPU. On CPU, falls back to torch.softmax with a warning.
    """
    if not _TRITON_AVAILABLE:
        raise ImportError(
            "triton is not installed. Install with: pip install triton\n"
            "Also requires a CUDA-capable GPU."
        )

    if not x.is_cuda:
        import warnings
        warnings.warn(
            "fused_softmax requires a CUDA tensor. Falling back to torch.softmax on CPU.",
            stacklevel=2,
        )
        return torch.softmax(x, dim=-1)

    # Flatten to 2D: [batch * seq_len, d_model] or [rows, cols]
    original_shape = x.shape
    if x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    x = x.contiguous()
    rows, cols = x.shape

    # Allocate output buffer
    output = torch.empty_like(x)

    # Choose BLOCK_SIZE as the next power-of-2 >= cols, capped at 4096.
    # Triton requires BLOCK_SIZE to be a compile-time constant (constexpr),
    # so we pick it at dispatch time based on input shape.
    BLOCK_SIZE = 1
    while BLOCK_SIZE < cols:
        BLOCK_SIZE *= 2
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)

    # Grid: one program per row
    grid = (rows,)

    softmax_kernel[grid](
        output,
        x,
        x.stride(0),      # stride between rows = cols (for contiguous layout)
        output.stride(0),
        cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.view(original_shape)
