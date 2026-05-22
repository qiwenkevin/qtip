import os
import torch
import qtip_kernels

CUSTOM_KERNELS = {
    K: {
        (2048, 2048): (
            getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_2048_1_2048"),
            getattr(qtip_kernels, f"decompress_matvec_custom_imma_16_{K}_2048_1_2048"),
        ),
        (512,  2048): (
            getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_512_1_2048"),
            getattr(qtip_kernels, f"decompress_matvec_custom_imma_16_{K}_512_1_2048"),
        ),
        (8192, 2048): (
            getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_8192_1_2048"),
            getattr(qtip_kernels, f"decompress_matvec_custom_imma_16_{K}_8192_1_2048"),
        ),
        (2048, 512): (
            getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_2048_1_512"),
            getattr(qtip_kernels, f"decompress_matvec_custom_imma_16_{K}_2048_1_512"),
        ),
        (2048, 8192): (
            getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_2048_1_8192"),
            getattr(qtip_kernels, f"decompress_matvec_custom_imma_16_{K}_2048_1_8192"),
        ),
    } for K in (2, 3, 4)
}

def prep(K, m, k, seed=0):
    g = torch.Generator(device='cuda').manual_seed(seed)
    compressed = torch.randint(-(2**31), 2**31 - 1,
                               (K * m * k // 32,),
                               dtype=torch.int32, generator=g, device='cuda')
    x_fp16 = (torch.randn((k,), generator=g, device='cuda', dtype=torch.float32) / 16
        ).clamp(-1, 1).to(torch.float16).reshape(k, 1)
    return compressed, x_fp16

def test_K(K):
    for (m, k), (fp16_fn, imma_fn) in CUSTOM_KERNELS[K].items():
        compressed, x_fp16 = prep(K, m, k)

        x_max = x_fp16.abs().max().to(torch.float32)
        x_scale = (x_max / 127.0).clamp_min(1e-30)
        x_int8 = (x_fp16.to(torch.float32) / x_scale).round().clamp(-127, 127).to(torch.int8)
        x_scale_t = x_scale.reshape(1).contiguous()
        out_imma = torch.zeros((m, 1), dtype=torch.float32, device='cuda')
        imma_fn(out_imma, compressed, x_int8, x_scale_t)
        torch.cuda.synchronize()

        x_dequant_fp16 = (x_int8.to(torch.float32) * x_scale).to(torch.float16)
        out_fp16 = torch.zeros((m, 1), dtype=torch.float32, device='cuda')
        fp16_fn(out_fp16, compressed, x_dequant_fp16)
        torch.cuda.synchronize()

        diff = (out_imma - out_fp16).abs()
        max_err = diff.max().item()
        ok = max_err < 0.5
        print(f"K={K} (m={m}, k={k}): max_err={max_err:.4f} ok={ok}")
        assert ok, (
            f"K={K} shape ({m},{k}) failed: out_imma diverges from out_fp16. "
            f"max diff={max_err:.4f}. First 5 IMMA: {out_imma[:5,0].tolist()}; "
            f"first 5 fp16: {out_fp16[:5,0].tolist()}"
        )

if __name__ == "__main__":
    only = int(os.environ.get("ONLY_K", "0"))
    for K in (2, 3, 4):
        if only and K != only: continue
        test_K(K)
    print("all custom IMMA kernels OK")
