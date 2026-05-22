import math
import os
import torch
import qtip_kernels
from lib.codebook.bitshift import decode_custom
from lib.utils.kernel_decompress import decode_compressed

def build_custom_lut(device='cuda'):
    lut = decode_custom(torch.arange(2**14, dtype=torch.int32, device=device))
    return lut.reshape(2**16, 1)  # int8

CUSTOM_KERNELS = {
    K: {
        (2048, 2048): getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_2048_1_2048"),
        (512,  2048): getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_512_1_2048"),
        (8192, 2048): getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_8192_1_2048"),
        (2048, 512):  getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_2048_1_512"),
        (2048, 8192): getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_2048_1_8192"),
    } for K in (2, 3, 4)
}

def prep(K, m, k, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    compressed = torch.randint(-(2**31), 2**31 - 1,
                               (K * m * k // 32,),
                               dtype=torch.int32, generator=g, device='cuda')
    x = (torch.randn((k,), generator=g, device='cuda', dtype=torch.float32) / 16
        ).clamp(-1, 1).to(torch.float16).reshape(k, 1)
    out = torch.zeros((m, 1), dtype=torch.float32, device='cuda')
    return out, compressed, x

def reference(K, m, k, compressed, x, lut):
    decompressed = decode_compressed(16, 9, K, 0, m, k, compressed, lut)
    return decompressed.to(torch.float16) @ x  # (m, 1) fp16

def test_K(K):
    lut = build_custom_lut()
    for (m, k), fn in CUSTOM_KERNELS[K].items():
        out, compressed, x = prep(K, m, k)
        fn(out, compressed, x)
        torch.cuda.synchronize()
        ref = reference(K, m, k, compressed, x, lut)
        ok = torch.allclose(out.half(), ref, atol=1e-3, rtol=0.01)
        max_err = (out.half() - ref).abs().max().item()
        print(f"K={K} (m={m}, k={k}): allclose={ok} max_err={max_err:.4f}")
        assert ok, f"K={K} shape ({m},{k}) failed"

if __name__ == "__main__":
    only = int(os.environ.get("ONLY_K", "0"))
    for K in (2, 3, 4):
        if only and K != only: continue
        torch._dynamo.reset()
        test_K(K)
    print("all custom kernels OK")
