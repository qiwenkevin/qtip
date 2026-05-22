# lib/utils/test_quant_activations.py
import torch
from lib.utils.quant_activations import quantize_act_int8

def test_roundtrip_relative_error_small():
    """Per-vector symmetric quant should reconstruct fp16 input within a
    bounded relative error.  The bound depends on the input distribution;
    for post-Hadamard activations (≈ Gaussian) on K=2048, single-vector
    int8 quant gives ~3e-3 RMSE / max-abs."""
    torch.manual_seed(0)
    x = torch.randn(2048, dtype=torch.float16, device='cuda') / 32.0
    x_int8, x_scale = quantize_act_int8(x)
    assert x_int8.dtype == torch.int8
    assert x_int8.shape == x.shape
    assert x_scale.dtype == torch.float32
    assert x_scale.shape == ()
    x_recon = x_int8.to(torch.float32) * x_scale
    rel_rmse = (x_recon.float() - x.float()).pow(2).mean().sqrt() / x.float().abs().max()
    assert rel_rmse < 5e-3, f"relative RMSE too high: {rel_rmse}"

def test_max_abs_bound():
    """Quantized values must be in [-127, 127]."""
    torch.manual_seed(1)
    x = torch.randn(4096, dtype=torch.float16, device='cuda')
    x_int8, _ = quantize_act_int8(x)
    assert x_int8.abs().max().item() <= 127

def test_all_zero_input_returns_zero_safe_scale():
    """If x is all zeros, scale should be a finite positive number and
    x_int8 should be all zeros — no NaN/inf."""
    x = torch.zeros(128, dtype=torch.float16, device='cuda')
    x_int8, x_scale = quantize_act_int8(x)
    assert x_int8.eq(0).all()
    assert torch.isfinite(x_scale).item()
    assert x_scale.item() > 0

if __name__ == "__main__":
    test_roundtrip_relative_error_small()
    test_max_abs_bound()
    test_all_zero_input_returns_zero_safe_scale()
    print("quant_activations OK")
