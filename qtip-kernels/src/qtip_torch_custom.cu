
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/types.h>
#include <torch/extension.h>
#include "inference_custom.cu"

using namespace torch::indexing;

template <uint32_t L, uint32_t R, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_matvec_custom(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x
) {
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.scalar_type() == torch::kFloat32);

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 1);
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);

    size_t m = out.size(0);
    size_t n = out.size(1);
    size_t k = x.size(0);

    TORCH_CHECK(m == M);
    TORCH_CHECK(k == K);
    TORCH_CHECK(compressed.numel() * 32 == R * m * k);
    TORCH_CHECK(x.size(1) == n);

    at::DeviceGuard guard(x.device());
    decompress_matvec_custom_ptr<L, R, M, N, K>(
            reinterpret_cast<float *>(out.data_ptr<float>()),
            reinterpret_cast<const uint32_t *>(compressed.data_ptr<int32_t>()),
            reinterpret_cast<const half2 *>(x.data_ptr<c10::Half>()),
            at::cuda::getCurrentCUDAStream()
    );
}

#define INSTANTIATE_CUSTOM(R, M, K)                                              \
__host__ extern void decompress_matvec_custom_16_##R##_##M##_1_##K(              \
        torch::Tensor &out,                                                      \
        torch::Tensor &compressed,                                               \
        torch::Tensor &x                                                         \
) {                                                                              \
    decompress_matvec_custom<16U, R##U, M##U, 1U, K##U>(out, compressed, x);     \
}

// K=2
INSTANTIATE_CUSTOM(2, 2048, 2048)
INSTANTIATE_CUSTOM(2, 512,  2048)
INSTANTIATE_CUSTOM(2, 8192, 2048)
INSTANTIATE_CUSTOM(2, 2048, 512)
INSTANTIATE_CUSTOM(2, 2048, 8192)

// K=3
INSTANTIATE_CUSTOM(3, 2048, 2048)
INSTANTIATE_CUSTOM(3, 512,  2048)
INSTANTIATE_CUSTOM(3, 8192, 2048)
INSTANTIATE_CUSTOM(3, 2048, 512)
INSTANTIATE_CUSTOM(3, 2048, 8192)

// K=4
INSTANTIATE_CUSTOM(4, 2048, 2048)
INSTANTIATE_CUSTOM(4, 512,  2048)
INSTANTIATE_CUSTOM(4, 8192, 2048)
INSTANTIATE_CUSTOM(4, 2048, 512)
INSTANTIATE_CUSTOM(4, 2048, 8192)

#undef INSTANTIATE_CUSTOM

