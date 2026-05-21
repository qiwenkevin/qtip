import qtip_kernels
import torch

kernels = [
    # LLama 3.2
    (2048, 1, 2048, 2),
    (512, 1, 2048, 2),
    (8192, 1, 2048, 2),
    (2048, 1, 512, 2),
    (2048, 1, 8192, 2),
    # Previous
    (1024, 1, 3072, 4),
    (8192, 1, 3072, 4),
    (3072, 1, 8192, 4),
    (3072, 1, 3072, 4),
    (53248, 1, 16384, 2),
    (53248, 1, 16384, 3),
    (53248, 1, 16384, 4),
    (16384, 1, 53248, 2),
    (16384, 1, 53248, 3),
    (16384, 1, 53248, 4),
    (1024, 1, 16384, 2),
    (1024, 1, 16384, 3),
    (1024, 1, 16384, 4),
    (16384, 1, 16384, 2),
    (16384, 1, 16384, 3),
    (16384, 1, 16384, 4),
    (4096, 1, 14336, 2),
    (4096, 1, 14336, 3),
    (4096, 1, 14336, 4),
    (14336, 1, 4096, 2),
    (14336, 1, 4096, 3),
    (14336, 1, 4096, 4),
    (1024, 1, 4096, 2),
    (1024, 1, 4096, 3),
    (1024, 1, 4096, 4),
    (4096, 1, 4096, 2),
    (4096, 1, 11008, 2),
    (11008, 1, 4096, 2),
    (12288, 1, 4096, 2),
    (22016, 1, 4096, 2),
    (8192, 1, 8192, 2),
    (10240, 1, 8192, 2),
    (10240, 1, 8192, 3),
    (10240, 1, 8192, 4),
    (57344, 1, 8192, 2),
    (57344, 1, 8192, 3),
    (57344, 1, 8192, 4),
    (8192, 1, 1024, 2),
    (8192, 1, 28672, 2),
    (28672, 1, 8192, 2),
    (1024, 1, 8192, 2),
    (4096, 1, 4096, 3),
    (4096, 1, 11008, 3),
    (11008, 1, 4096, 3),
    (12288, 1, 4096, 3),
    (22016, 1, 4096, 3),
    (8192, 1, 8192, 3),
    (8192, 1, 1024, 3),
    (8192, 1, 28672, 3),
    (28672, 1, 8192, 3),
    (1024, 1, 8192, 3),
    (4096, 1, 4096, 4),
    (4096, 1, 11008, 4),
    (11008, 1, 4096, 4),
    (12288, 1, 4096, 4),
    (22016, 1, 4096, 4),
    (8192, 1, 8192, 4),
    (8192, 1, 1024, 4),
    (8192, 1, 28672, 4),
    (28672, 1, 8192, 4),
    (1024, 1, 8192, 4),
]

kdict = {}

for m, n, k, bitrate in kernels:
    torch.library.define(
        f"quip_lib::decompress_matvec_qtip_{m}_{n}_{k}_{bitrate}",
        "(Tensor compressed, Tensor x, Tensor codebook) -> Tensor")

    name = f"decompress_matvec_qtip_{m}_{n}_{k}_{bitrate}"
    kernel_name = f"qtip_kernels.decompress_matvec_16_9_{bitrate}_1_{m}_{n}_{k}"
    exec(f"""\
@torch.library.register_fake("quip_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, {m}, dtype=torch.float32, device=x.device)

@torch.library.impl("quip_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({m}, 1), dtype=torch.float32, device=x.device)
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), x.to(torch.float16).T, codebook.reshape(-1))
    return out.T
    """)


# Custom-decode kernels (V=1, no codebook arg in the op).
custom_kernels = [
    (m, 1, k, K_)
    for K_ in (2, 3, 4)
    for (m, k) in (
        (2048, 2048),
        (512,  2048),
        (8192, 2048),
        (2048, 512),
        (2048, 8192),
    )
]

for m, n, k, bitrate in custom_kernels:
    torch.library.define(
        f"quip_lib::decompress_matvec_qtip_custom_{m}_{n}_{k}_{bitrate}",
        "(Tensor compressed, Tensor x) -> Tensor")

    name        = f"decompress_matvec_qtip_custom_{m}_{n}_{k}_{bitrate}"
    kernel_name = f"qtip_kernels.decompress_matvec_custom_16_{bitrate}_{m}_{n}_{k}"
    exec(f"""\
@torch.library.register_fake("quip_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, {m}, dtype=torch.float32, device=x.device)

@torch.library.impl("quip_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({m}, 1), dtype=torch.float32, device=x.device)
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), x.to(torch.float16).T)
    return out.T
    """)


# Custom-decode int8 IMMA kernels (V=1, no codebook arg; int8 activations + scale).
# Same shapes as custom_kernels, different op name and signature.
custom_imma_kernels = [
    (m, 1, k, K_)
    for K_ in (2, 3, 4)
    for (m, k) in (
        (2048, 2048),
        (512,  2048),
        (8192, 2048),
        (2048, 512),
        (2048, 8192),
    )
]

for m, n, k, bitrate in custom_imma_kernels:
    torch.library.define(
        f"quip_lib::decompress_matvec_qtip_custom_imma_{m}_{n}_{k}_{bitrate}",
        "(Tensor compressed, Tensor x_int8, Tensor x_scale) -> Tensor")

    name        = f"decompress_matvec_qtip_custom_imma_{m}_{n}_{k}_{bitrate}"
    kernel_name = f"qtip_kernels.decompress_matvec_custom_imma_16_{bitrate}_{m}_{n}_{k}"
    exec(f"""\
@torch.library.register_fake("quip_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        x_int8: torch.Tensor,
        x_scale: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, {m}, dtype=torch.float32, device=x_int8.device)

@torch.library.impl("quip_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        x_int8: torch.Tensor,
        x_scale: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({m}, 1), dtype=torch.float32, device=x_int8.device)
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), x_int8, x_scale)
    return out.T
    """)
