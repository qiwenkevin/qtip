# lib/utils/quant_activations.py
"""Per-vector symmetric int8 quantization for fp16 activations.

Used by the int8 IMMA custom-decode kernel to convert fp16 activations to
int8 before the IMMA matvec. Single scalar fp32 scale per call; the kernel
re-applies the scale to its int32 accumulator at output time.
"""
import torch

_MIN_SCALE = 1.0e-30


def quantize_act_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp16 (or fp32) `x` to int8 with per-vector symmetric scale.

    Returns (x_int8, x_scale_fp32) such that x ≈ x_int8.float() * x_scale_fp32.
    Caller must keep `x_scale_fp32` alongside `x_int8` and pass both to the
    kernel.
    """
    assert x.is_cuda
    x_max = x.detach().abs().max().to(torch.float32)
    x_scale = (x_max / 127.0).clamp_min(_MIN_SCALE)
    x_int8 = (x.to(torch.float32) / x_scale).round().clamp(-127, 127).to(torch.int8)
    return x_int8.contiguous(), x_scale
