import torch
import qtip_kernels
import time

SHAPES = [(2048, 2048), (512, 2048), (8192, 2048), (2048, 512), (2048, 8192)]
WARMUP = 50
ITERS  = 200

def time_call(fn, args, sync=True):
    if sync: torch.cuda.synchronize()
    for _ in range(WARMUP): fn(*args)
    if sync: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS): fn(*args)
    if sync: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e6  # µs

def main():
    torch.manual_seed(0)
    print(f"{'K':>2} {'m':>5} {'k':>5}  {'fp16 µs':>9} {'imma µs':>9}  {'speedup':>7}")
    for K in (2, 3, 4):
        for (m, k) in SHAPES:
            compressed = torch.randint(-(2**31), 2**31 - 1,
                                       (K * m * k // 32,),
                                       dtype=torch.int32, device='cuda')
            x_fp16 = torch.randn((k, 1), dtype=torch.float16, device='cuda')
            x_int8 = torch.randint(-127, 127, (k, 1), dtype=torch.int8, device='cuda')
            x_scale = torch.tensor([1e-3], dtype=torch.float32, device='cuda')
            out = torch.zeros((m, 1), dtype=torch.float32, device='cuda')

            fp16_fn = getattr(qtip_kernels, f"decompress_matvec_custom_16_{K}_{m}_1_{k}")
            imma_fn = getattr(qtip_kernels, f"decompress_matvec_custom_imma_16_{K}_{m}_1_{k}")

            t_fp16 = time_call(fp16_fn, (out, compressed, x_fp16))
            t_imma = time_call(imma_fn, (out, compressed, x_int8, x_scale))
            print(f"{K:>2} {m:>5} {k:>5}  {t_fp16:>9.2f} {t_imma:>9.2f}  {t_fp16/t_imma:>7.2f}x")

if __name__ == "__main__":
    main()
