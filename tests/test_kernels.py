"""
Correctness tests for all Triton kernels.

Tests are skipped gracefully if triton or torch are not installed,
or if no CUDA GPU is available.
"""

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

# Skip all tests if no CUDA GPU
if not torch.cuda.is_available():
    pytest.skip("No CUDA GPU available, skipping kernel tests", allow_module_level=True)

from kernels.softmax import fused_softmax
from kernels.elementwise import triton_relu, triton_gelu
from kernels.matrix_ops import triton_matmul


# ── Softmax tests ──────────────────────────────────────────────────────────────

def test_softmax_sums_to_one_per_row():
    """Each row of softmax output must sum to 1.0 (within float32 tolerance)."""
    x = torch.randn(64, 256, device="cuda", dtype=torch.float32)
    out = fused_softmax(x)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(64, device="cuda"), atol=1e-5), \
        f"Row sums not close to 1.0: max_err={((row_sums - 1.0).abs().max()):.2e}"


def test_softmax_handles_batch_dimension():
    """Softmax should work on 3D inputs [batch, seq_len, d_model]."""
    x = torch.randn(4, 16, 128, device="cuda", dtype=torch.float32)
    out = fused_softmax(x)
    assert out.shape == x.shape
    # Check last-dim sums
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_softmax_handles_single_element():
    """A 1-element row should softmax to exactly 1.0."""
    x = torch.tensor([[3.5]], device="cuda", dtype=torch.float32)
    out = fused_softmax(x)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


# ── ReLU tests ─────────────────────────────────────────────────────────────────

def test_relu_zeros_negatives():
    """All negative input values should become 0, positive values unchanged."""
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 5.0], device="cuda")
    out = triton_relu(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 5.0], device="cuda")
    assert torch.allclose(out, expected, atol=1e-6)


def test_elementwise_handle_non_contiguous_tensors():
    """Triton ops should handle non-contiguous tensors (via .contiguous() internally)."""
    x = torch.randn(16, 16, device="cuda")
    x_non_contig = x.T  # transposed is non-contiguous
    out = triton_relu(x_non_contig)
    expected = torch.relu(x_non_contig)
    assert torch.allclose(out, expected, atol=1e-6)


# ── GELU tests ─────────────────────────────────────────────────────────────────

def test_gelu_output_shape_matches_input():
    """GELU output shape must match input shape exactly."""
    x = torch.randn(32, 512, device="cuda")
    out = triton_gelu(x)
    assert out.shape == x.shape


def test_output_dtype_preserved():
    """Output dtype must match input dtype (float32 in, float32 out)."""
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    out_relu = triton_relu(x)
    out_gelu = triton_gelu(x)
    assert out_relu.dtype == torch.float32
    assert out_gelu.dtype == torch.float32


# ── Matmul tests ───────────────────────────────────────────────────────────────

def test_matmul_output_matches_torch_matmul():
    """Custom Triton matmul must match torch.matmul within float32 tolerance."""
    torch.manual_seed(42)
    a = torch.randn(256, 128, device="cuda", dtype=torch.float32)
    b = torch.randn(128, 256, device="cuda", dtype=torch.float32)

    c_torch = torch.matmul(a, b)
    c_triton = triton_matmul(a, b)

    assert c_triton.shape == c_torch.shape
    assert torch.allclose(c_torch, c_triton, atol=1e-3), \
        f"Max diff: {(c_torch - c_triton).abs().max():.4e}"
