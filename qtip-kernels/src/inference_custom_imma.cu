
#include <cstdio>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <mma.h>
#include <c10/cuda/CUDAStream.h>

#include "inference.h"

using namespace nvcuda;

#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)      do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (false)

#define BLOCKS_PER_SM 2
#define MMA_M         16
#define MMA_N         8
#define MMA_K         16
#define BLOCK_COUNT   256
#define WARP_SIZE     32
#define BLOCK_SIZE    512
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define FULL_MASK     0xFFFFFFFFU


__inline__ __device__ uint32_t ld_cs(const uint32_t* p)
{
    uint32_t out;
    asm("ld.global.cs.u32 %0, [%1];" : "=r"(out) : "l"(p));
    return out;
}
__inline__ __device__ uint2 ld_cs(const uint2* p)
{
    uint2 out;
    asm("ld.global.cs.v2.u32 {%0, %1}, [%2];" : "=r"(out.x), "=r"(out.y) : "l"(p));
    return out;
}
__inline__ __device__ uint3 ld_cs(const uint3* p)
{
    uint3 out;
    asm("ld.global.cs.u32 %0, [%1];"    : "=r"(out.x) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+4];"  : "=r"(out.y) : "l"(p));
    asm("ld.global.cs.u32 %0, [%1+8];"  : "=r"(out.z) : "l"(p));
    return out;
}
__inline__ __device__ uint4 ld_cs(const uint4* p)
{
    uint4 out;
    asm("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(p));
    return out;
}
__inline__ __device__ void prefetch(uint32_t *a){
    asm("prefetch.global.L1 [%0];"::"l"(a));
}

#define LD_CS
template <uint32_t R>
__device__ inline void load_reg_cs(const uint16_t *__restrict__ compressed,
                                   int weight_idx, uint32_t laneId,
                                   uint4 &reg_cs_next, uint4 &reg_cs2_next) {
    if constexpr(R == 2) {
        ditto2 reg_load = {.u32x2 = ld_cs((uint2 *) &compressed[weight_idx])};
        uint32_t next1 = __shfl_sync(FULL_MASK, reg_load.u32x2.x, laneId + 1);
        uint32_t next2 = __shfl_sync(FULL_MASK, reg_load.u32x2.y, laneId + 1);
        reg_cs_next.x = __byte_perm(next1, reg_load.u32x2.x, 0x5410);
        reg_cs_next.y = __byte_perm(next1, reg_load.u32x2.x, 0x7632);
        reg_cs_next.z = __byte_perm(next2, reg_load.u32x2.y, 0x5410);
        reg_cs_next.w = __byte_perm(next2, reg_load.u32x2.y, 0x7632);
    } else if constexpr(R == 3) {
        uint3 reg_load = ld_cs((uint3 *) &compressed[weight_idx]);
        uint32_t reg_load1 = reg_load.x, reg_load2 = reg_load.y, reg_load3 = reg_load.z;
        uint32_t reg_24_1 = reg_load1 & 0xffffff;
        uint32_t reg_24_2 = ((reg_load1 >> 24) | (reg_load2 << 8)) & 0xffffff;
        uint32_t reg_24_3 = ((reg_load2 >> 16) | (reg_load3 << 16)) & 0xffffff;
        uint32_t reg_24_4 = (reg_load3 >> 8) & 0xffffff;
        uint32_t pack1 = (reg_24_1 >> 8) | ((reg_24_2 << 8) & 0xffff0000);
        uint32_t pack3 = (reg_24_3 >> 8) | ((reg_24_4 << 8) & 0xffff0000);
        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);
        reg_cs_next.x = __byte_perm(next1, reg_24_1, 0x6541);
        reg_cs_next.y = __byte_perm(next1, reg_24_2, 0x6543);
        reg_cs_next.z = __byte_perm(next3, reg_24_3, 0x6541);
        reg_cs_next.w = __byte_perm(next3, reg_24_4, 0x6543);
        reg_cs2_next.x = ((next1 >> 3)  & 0x1FFF) | (reg_24_1 << 13);
        reg_cs2_next.y = ((next1 >> 19) & 0x1FFF) | (reg_24_2 << 13);
        reg_cs2_next.z = ((next3 >> 3)  & 0x1FFF) | (reg_24_3 << 13);
        reg_cs2_next.w = ((next3 >> 19) & 0x1FFF) | (reg_24_4 << 13);
    } else if constexpr(R == 4) {
        uint4 reg_load = ld_cs((uint4 *) &compressed[weight_idx]);
        uint32_t reg_load1 = reg_load.x, reg_load2 = reg_load.y, reg_load3 = reg_load.z, reg_load4 = reg_load.w;
        uint32_t pack1 = (reg_load1 >> 16) | (reg_load2 & 0xffff0000);
        uint32_t pack3 = (reg_load3 >> 16) | (reg_load4 & 0xffff0000);
        uint32_t next1 = __shfl_sync(FULL_MASK, pack1, laneId + 1);
        uint32_t next3 = __shfl_sync(FULL_MASK, pack3, laneId + 1);
        reg_cs_next.x = reg_load1;
        reg_cs_next.y = reg_load2;
        reg_cs_next.z = reg_load3;
        reg_cs_next.w = reg_load4;
        reg_cs2_next.x = ((reg_load1 & 0xFFFu) << 12) | ((next1 >>  4) & 0xFFFu);
        reg_cs2_next.y = ((reg_load2 & 0xFFFu) << 12) | ((next1 >> 20) & 0xFFFu);
        reg_cs2_next.z = ((reg_load3 & 0xFFFu) << 12) | ((next3 >>  4) & 0xFFFu);
        reg_cs2_next.w = ((reg_load4 & 0xFFFu) << 12) | ((next3 >> 20) & 0xFFFu);
    }
}

__device__ inline int8_t decode_custom_one(uint32_t state16) {
    uint32_t x   = state16 >> 2;
    uint32_t sub = state16 & 3u;
    uint32_t m   = 0x3F3F3F3Fu;
    uint32_t u1  = x * 34038481u + 76625530u;
    uint32_t u2  = x * 88827277u + 46632450u;
    uint32_t u3  = x * 53179724u + 16848693u;
    uint32_t u4  = x * 60450533u + 92801199u;
    uint32_t y   = ((u1 & m) + (u2 & m) + (u3 & m) + (u4 & m)
                    + 0x02020202u) ^ 0x80808080u;
    return (int8_t)((y >> (sub * 8u)) & 0xFFu);
}

template <uint32_t L, uint32_t R, uint32_t M, uint32_t N, uint32_t K>
__global__ static void
__launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
kernel_decompress_matvec_custom_imma(
    float *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const int8_t *__restrict__ x_int8,
    const float *__restrict__ x_scale_ptr
) {
    float x_scale = *x_scale_ptr;
    uint32_t threadId = threadIdx.x;
    uint32_t laneId = threadIdx.x % WARP_SIZE;
    uint32_t warpId = threadId / WARP_SIZE;
    uint32_t blockId = blockIdx.x;

    constexpr uint32_t tileCountM = M / MMA_M;
    constexpr uint32_t tileCountK = K / MMA_K;

    constexpr uint32_t warps_per_block = BLOCK_SIZE / WARP_SIZE;

#define ROUND_UP(a, b) ((a + b - 1) / b)

    static_assert (tileCountM % 2 == 0);
    constexpr uint32_t m_per_block = ROUND_UP(tileCountM, (2 * BLOCK_COUNT));
    constexpr uint32_t k_per_block = tileCountK / (warps_per_block * 4) * 2;
    static_assert((tileCountK % (warps_per_block * 4)) % 4 == 0);
    uint32_t this_warp_k = (warpId < (tileCountK % (warps_per_block * 4)) / 4) ? k_per_block + 2 : k_per_block;

    constexpr uint32_t u16_per_compressed_tile = MMA_M * MMA_K * R / 16;
    constexpr uint32_t f32_per_out_tile = MMA_M;

    uint32_t tileIdM = m_per_block * blockId;

    constexpr uint32_t weight_block = 4;
    constexpr uint32_t u16_per_tile_block = u16_per_compressed_tile * weight_block;
    constexpr uint32_t weight_step = warps_per_block * u16_per_tile_block;
    constexpr uint32_t weight_row_step = tileCountK * u16_per_compressed_tile * 2;


    for (uint32_t mi = 0; mi < m_per_block; mi+=1) {
        if (tileIdM * 2 >= tileCountM) return;
        int weight_idx = tileIdM * weight_row_step + warpId * u16_per_tile_block * 2 + laneId * (u16_per_tile_block / WARP_SIZE);
        uint4 reg_cs_next = {};
        uint4 reg_cs2_next = {};
        load_reg_cs<R>((const uint16_t * __restrict__) compressed, weight_idx, laneId, reg_cs_next, reg_cs2_next);
        uint4 reg_cs;
        uint4 reg_cs2;

        int4 reg_p[2] = {};

        __shared__ uint32_t x_buf[BLOCK_SIZE / WARP_SIZE][4][4];

        constexpr uint32_t u32_per_kfill = 16;  // 4 K-tiles * 4 uint32/K-tile
        uint32_t x_uidx = warpId * u32_per_kfill + laneId;
        constexpr uint32_t x_uidx_step = warps_per_block * u32_per_kfill;

#pragma unroll 4
        for (uint32_t ki = 0; ki < this_warp_k; ki += 1) {
            // load next 2x2 block of weight tiles
            if (ki + 1 != this_warp_k && ki % 2 == 1) weight_idx += weight_step * 2;
            reg_cs = reg_cs_next;
            reg_cs2 = reg_cs2_next;
            load_reg_cs<R>((const uint16_t * __restrict__) compressed,
                            weight_idx + (1 - ki % 2) * u16_per_tile_block,
                            laneId, reg_cs_next, reg_cs2_next);

            if (ki % 2 == 0) {
                __syncwarp();
                if (laneId < u32_per_kfill) {
                    uint32_t buf_pos  = laneId / 4;   // ki-position (0..3 = ki,ki+1's 4 subki)
                    uint32_t buf_lane = laneId % 4;   // lane (0..3)
                    x_buf[warpId][buf_pos][buf_lane] =
                        *(reinterpret_cast<const uint32_t *>(x_int8) + x_uidx);
                }
                __syncwarp();
                x_uidx += x_uidx_step;
            }

#pragma unroll 2
            for (uint32_t subki = 0; subki < 2; subki += 1) {
                uint32_t reg_a = 0;
                if (laneId < 4) {
                    reg_a = x_buf[warpId][ki % 2 * 2 + subki][laneId];
                }

                const uint32_t src_lane_lo = (laneId & ~3u) | ((laneId & 1u) << 1);
                const uint32_t src_lane_hi = src_lane_lo | 1u;
                const bool     use_high_half = (laneId & 2u) != 0;

#pragma unroll 2
                for (uint32_t submi = 0; submi < 2; submi++) {
                    uint32_t reg_c, reg_c2;
                    if (submi == 0 && subki == 0) reg_c = reg_cs.x;
                    else if (submi == 1 && subki == 0) reg_c = reg_cs.y;
                    else if (submi == 0 && subki == 1) reg_c = reg_cs.z;
                    else if (submi == 1 && subki == 1) reg_c = reg_cs.w;
                    if (submi == 0 && subki == 0) reg_c2 = reg_cs2.x;
                    else if (submi == 1 && subki == 0) reg_c2 = reg_cs2.y;
                    else if (submi == 0 && subki == 1) reg_c2 = reg_cs2.z;
                    else if (submi == 1 && subki == 1) reg_c2 = reg_cs2.w;
                    (void)reg_c2;

                    int8_t w[8];
                    #pragma unroll
                    for (uint32_t j = 0; j < 8; ++j) {
                        uint32_t state16;
                        if constexpr(R == 2) {
                            state16 = (reg_c >> (16u - 2u * j)) & 0xFFFFu;
                        } else if constexpr(R == 3) {
                            if (j < 6) {
                                state16 = (reg_c  >> (16u - 3u * j)) & 0xFFFFu;
                            } else {
                                state16 = (reg_c2 >> (21u - 3u * j)) & 0xFFFFu;
                            }
                        } else if constexpr(R == 4) {
                            if (j < 5) {
                                state16 = (reg_c  >> (16u - 4u * j)) & 0xFFFFu;
                            } else {
                                state16 = (reg_c2 >> (28u - 4u * j)) & 0xFFFFu;
                            }
                        }
                        w[j] = decode_custom_one(state16);
                    }

                    uint32_t my_low_row =
                          (((uint32_t)(uint8_t)w[0])      )
                        | (((uint32_t)(uint8_t)w[1]) <<  8)
                        | (((uint32_t)(uint8_t)w[4]) << 16)
                        | (((uint32_t)(uint8_t)w[5]) << 24);
                    uint32_t my_high_row =
                          (((uint32_t)(uint8_t)w[2])      )
                        | (((uint32_t)(uint8_t)w[3]) <<  8)
                        | (((uint32_t)(uint8_t)w[6]) << 16)
                        | (((uint32_t)(uint8_t)w[7]) << 24);

                    uint32_t recv_low_a  = __shfl_sync(FULL_MASK, my_low_row,  src_lane_lo);
                    uint32_t recv_low_b  = __shfl_sync(FULL_MASK, my_low_row,  src_lane_hi);
                    uint32_t recv_high_a = __shfl_sync(FULL_MASK, my_high_row, src_lane_lo);
                    uint32_t recv_high_b = __shfl_sync(FULL_MASK, my_high_row, src_lane_hi);

                    const uint32_t bp_sel = use_high_half ? 0x7632u : 0x5410u;

                    ditto2 reg_w;
                    reg_w.u32[0] = __byte_perm(recv_low_a,  recv_low_b,  bp_sel);
                    reg_w.u32[1] = __byte_perm(recv_high_a, recv_high_b, bp_sel);

                    asm volatile (
                        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
                        " {%0, %1, %2, %3},"
                        " {%4, %5},"
                        " {%6},"
                        " {%0, %1, %2, %3};"
                        : "+r"(reg_p[submi].x), "+r"(reg_p[submi].y),
                          "+r"(reg_p[submi].z), "+r"(reg_p[submi].w)
                        :  "r"(reg_w.u32[0]), "r"(reg_w.u32[1]),
                           "r"(reg_a)
                    );
                }
            }

            if (ki % 2 == 0) {
                // Prefetch the next next pair's x_int8 region.
                prefetch((uint32_t *) x_int8 + x_uidx + x_uidx_step);
            }
        }

        // ** reduce + write fp32 = int32_acc * x_scale **
        __shared__ __align__(16 * 8*32) int32_t reduce_gather[BLOCK_SIZE / WARP_SIZE][2][16];
        if (laneId % 4 == 0) {
            for (int pi = 0; pi < 2; pi++) {
                reduce_gather[warpId][pi][laneId / 4]      = reg_p[pi].x;
                reduce_gather[warpId][pi][laneId / 4 + 8]  = reg_p[pi].z;
            }
        }
        __syncthreads();

        if (warpId < 1) {
            int pi = laneId / 16;
            // int32 is sufficient: per-warp slot is bounded by
            //   |int8|^2 * MMA_K * this_warp_k * 2  (two submi per slot)
            //   = 127*127*16 * (this_warp_k*2)
            // Summed across BLOCK_SIZE/WARP_SIZE=16 warps. For the shapes we
            // ship (K<=8192, this_warp_k<=8), worst case is ~66M << 2^31.
            // Bench shapes assert this; revisit if K>=131072 is ever added.
            int32_t reduced = 0;
            for (int warpi = 0; warpi < BLOCK_SIZE / WARP_SIZE; warpi++) {
                reduced += reduce_gather[warpi][pi][laneId % 16];
            }

            float *out_tile = out + (tileIdM * 2) * f32_per_out_tile;
            out_tile[laneId] = ((float)reduced) * x_scale;
        }
        if constexpr(m_per_block > 1) __syncthreads();
        tileIdM += 1;
    }
}

template <uint32_t L, uint32_t R, uint32_t M, uint32_t N, uint32_t K>
__host__ static void decompress_matvec_custom_imma_ptr(
    float *__restrict__ out,
    const uint32_t *__restrict__ compressed,
    const int8_t *__restrict__ x_int8,
    const float *__restrict__ x_scale_ptr,
    CUstream_st *stream
) {
    static_assert(L == 16, "Only L=16 supported");
    static_assert(R == 2 || R == 3 || R == 4, "Bitrate must be 2, 3, or 4");
    static_assert(M % MMA_M == 0);
    static_assert(N == 1);
    static_assert(K % MMA_K == 0);
    static_assert(BLOCK_SIZE % WARP_SIZE == 0);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    assert(deviceProp.warpSize == WARP_SIZE);

    constexpr uint32_t gridSize  = BLOCK_COUNT;
    constexpr uint32_t blockSize = BLOCK_SIZE;
    kernel_decompress_matvec_custom_imma<L, R, M, N, K>
        <<<gridSize, blockSize, 0, stream>>>(out, compressed, x_int8, x_scale_ptr);
    gpuErrchk(cudaPeekAtLastError());
}
