"""
Tiled matrix multiplication kernel in Triton.

Why tiling matters:
--------------------
Naive matmul (A @ B = C) for an [M, K] × [K, N] multiplication requires
reading M*K + K*N floats from global memory (HBM) and writing M*N results.
If we compute each C[i,j] independently, we re-read entire rows/columns
of A and B repeatedly: terrible bandwidth utilization.

Tiling (also called blocking) works around this by loading a BLOCK_M × BLOCK_K
tile of A and a BLOCK_K × BLOCK_N tile of B into fast SRAM (shared memory /
L2 cache), computing the partial dot product, then advancing to the next tile
along the K dimension. Each tile is loaded once and reused for all output
elements in the corresponding output tile.

This reduces global memory traffic from O(MNK) to O(MK + KN + MN), the
same asymptotic bound but with much better constant factors due to cache reuse.

Register tiling vs shared memory tiling:
  Triton's programming model exposes tile-level parallelism at the register
  level. Each program instance operates on a BLOCK_M × BLOCK_N output tile
  and accumulates it fully before writing, which is exactly what cuBLAS does
  at its core.

Requires: CUDA GPU, triton>=2.0, torch>=2.0
"""

try:
    import triton
    import triton.language as tl
    import torch
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def matmul_kernel(
        a_ptr,   # pointer to A matrix [M, K]
        b_ptr,   # pointer to B matrix [K, N]
        c_ptr,   # pointer to C matrix [M, N]
        M, N, K,            # matrix dimensions
        stride_am, stride_ak,  # A's row stride, column stride
        stride_bk, stride_bn,  # B's row stride, column stride
        stride_cm, stride_cn,  # C's row stride, column stride
        BLOCK_M: tl.constexpr,   # tile height in M dimension
        BLOCK_N: tl.constexpr,   # tile width in N dimension
        BLOCK_K: tl.constexpr,   # tile depth in K (reduction) dimension
    ):
        """
        Each program instance computes one BLOCK_M × BLOCK_N tile of C.

        The 2D grid is laid out as:
          grid_m = ceil(M / BLOCK_M)
          grid_n = ceil(N / BLOCK_N)
          total programs = grid_m * grid_n

        We use program_id(0) to identify the (m_tile, n_tile) position via
        a super-group ordering that improves L2 cache hit rate.
        """
        # ── Map program_id to (m_tile, n_tile) coordinates ────────────────────
        pid = tl.program_id(axis=0)

        # Number of tiles in the N dimension
        num_pid_n = tl.cdiv(N, BLOCK_N)

        # Simple row-major ordering: pid → (pid // num_pid_n, pid % num_pid_n)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        # ── Compute starting offsets for this tile ─────────────────────────────
        # Row indices for the M-tile [BLOCK_M,]
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # Column indices for the N-tile [BLOCK_N,]
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # K-dimension offsets for the inner loop [BLOCK_K,]
        offs_k = tl.arange(0, BLOCK_K)

        # ── Pointers to the first K-tile of A and B ────────────────────────────
        # a_ptrs shape: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # b_ptrs shape: [BLOCK_K, BLOCK_N]
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # ── Accumulator: holds partial dot products for the output tile ────────
        # Initialize to zero; dtype must match output precision
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ── Main K-loop: iterate over tiles along the K dimension ──────────────
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            # Mask for the K boundary (last tile may be partial)
            k_mask = offs_k < K - k * BLOCK_K

            # Load A tile: [BLOCK_M, BLOCK_K]
            a = tl.load(
                a_ptrs,
                mask=k_mask[None, :],   # broadcast over M
                other=0.0,
            )
            # Load B tile: [BLOCK_K, BLOCK_N]
            b = tl.load(
                b_ptrs,
                mask=k_mask[:, None],   # broadcast over N
                other=0.0,
            )

            # ── Dot product for this K-tile: accumulate into acc ───────────────
            # tl.dot is the Triton equivalent of a matrix-multiply accumulate (MMA)
            # instruction. On Ampere/Hopper GPUs this maps to Tensor Core ops.
            acc += tl.dot(a, b)

            # ── Advance pointers to the next K-tile ───────────────────────────
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # ── Cast accumulator to output dtype and store ─────────────────────────
        c = acc.to(tl.float32)

        # Compute output pointers [BLOCK_M, BLOCK_N]
        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

        # Boundary masks for M and N
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """
    Matrix multiplication via custom tiled Triton kernel.

    Args:
        a: Float32 CUDA tensor of shape [M, K].
        b: Float32 CUDA tensor of shape [K, N].

    Returns:
        Float32 CUDA tensor of shape [M, N].

    Requires:
        CUDA GPU. Falls back to torch.matmul with a warning on CPU.
    """
    if not _TRITON_AVAILABLE:
        raise ImportError("triton is not installed. pip install triton")

    assert a.ndim == 2 and b.ndim == 2, "Only 2D matrix multiplication is supported"
    assert a.shape[1] == b.shape[0], f"Shape mismatch: A is {a.shape}, B is {b.shape}"

    if not a.is_cuda or not b.is_cuda:
        import warnings
        warnings.warn("triton_matmul requires CUDA tensors. Falling back to torch.matmul.", stacklevel=2)
        return torch.matmul(a, b)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Tile sizes: these are autotunable; 128x128x32 is a good starting point
    # for sizes that fit in L2. Triton's autotune would sweep these.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c
