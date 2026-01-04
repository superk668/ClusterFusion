/**
 * ClusterFusion Pythia-2.8B MLP Down Kernel (Split Version)
 * 
 * Handles: MLP Down Projection → Cluster Reduce → Final Residual Connection
 * 
 * Inputs:
 *   - mlp_intermediate: [FFN_DIM] - MLP up output after GELU (from attention kernel)
 *   - input: [1, HIDDEN_DIM] - original input for residual
 *   - attn_output: [1, HIDDEN_DIM] - attention output for residual
 * 
 * Output:
 *   - output: [1, HIDDEN_DIM] = input + attn_output + mlp_output
 * 
 * Key difference from fused kernel:
 *   - No grid.sync() needed (replaced by kernel launch boundary)
 *   - Uses regular kernel launch (not cooperative)
 */

#include <cuda/barrier>
#include <cudaTypedefs.h>
#include <math_constants.h> 
#include "../../dsm.cuh"
#include "config.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(CLUSTER_SIZE, 1, 1) PythiaMlpDownKernel(
    // Output tensor
    half* output,           // [1, HIDDEN_DIM] - final output
    // Input tensors
    half* input,            // [1, HIDDEN_DIM] - original input
    half* attn_output,      // [1, HIDDEN_DIM] - attention output
    half* mlp_intermediate, // [FFN_DIM] - MLP up output after GELU
    // MLP weights
    half* mlp_down_bias,    // [HIDDEN_DIM]
    // TMA descriptor
    const __grid_constant__ CUtensorMap tensor_map_mlp_down
) {
    // ==================== Thread/Block Indexing ====================
    // Note: No grid group needed - we don't use grid.sync()
    // Kernel launch boundary provides synchronization from attention kernel
    cg::cluster_group cluster       = cg::this_cluster();
    cg::thread_block block          = cg::this_thread_block();
    const uint32_t head_id          = blockIdx.x / CLUSTER_SIZE;
    const uint32_t cluster_block_id = cluster.block_rank();
    const uint32_t tid              = block.thread_rank();
    const uint32_t lane_id          = tid % WARP_SIZE;
    const uint32_t warp_id          = tid / WARP_SIZE;

    // ==================== Shared Memory Layout ====================
    extern __shared__ uint8_t shmem_base[];
    half* weight      = reinterpret_cast<half*>((uintptr_t)(shmem_base) + 127 & ~127);
    half* local_qkv   = weight + 2 * TMA_LOAD_ONCE * MAX_SMEM_DIM;

    // ==================== Register Allocation ====================
    float tmp = 0.0f;
    half __align__(16) reg_input[NUM_PER_THREAD];
    uint32_t size, src_addr, dst_addr, neighbor_dst_bar = 0;

    // ==================== Barrier Initialization ====================
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[2];
    barrier::arrival_token token[2];
    __shared__ uint64_t barrier_mem;
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier_mem));
    if (tid == 0) {
        init(&bar[0], blockDim.x);
        cde::fence_proxy_async_shared_cta();
        init(&bar[1], blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    block.sync();

    // ==================== Precompute Thread Indices ====================
    uint input_idx   = (lane_id % NUM_THREAD_PER_ROW) * NUM_PER_THREAD;
    uint weight_idx  = warp_id * NUM_ROW_PER_WARP + lane_id / NUM_THREAD_PER_ROW;

    // ==================== MLP Down Projection ====================
    uint32_t down_out_offset = head_id * HEAD_DIM;
    uint32_t ffn_input_offset = cluster_block_id * (FFN_DIM / CLUSTER_SIZE);
    uint32_t ffn_input_per_block = FFN_DIM / CLUSTER_SIZE;

    tmp = 0.0f;
    if (tid == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[0], &tensor_map_mlp_down, ffn_input_offset, down_out_offset, bar[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, TMA_LOAD_ONCE_SIZE);
    } else {
        token[0] = bar[0].arrive();
    }

    half __align__(16) mlp_in[NUM_PER_THREAD];
    for (int id = 1; id < ffn_input_per_block / TMA_LOAD_ONCE; id++) {
        if (tid == 0) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(&weight[(id % 2) * TMA_LOAD_ONCE_NUM], &tensor_map_mlp_down, ffn_input_offset + id * TMA_LOAD_ONCE, down_out_offset, bar[id % 2]);
            token[id % 2] = cuda::device::barrier_arrive_tx(bar[id % 2], 1, TMA_LOAD_ONCE_SIZE);
        } else {
            token[id % 2] = bar[id % 2].arrive();
        }
        bar[(id - 1) % 2].wait(std::move(token[(id - 1) % 2]));
        if (weight_idx < HEAD_DIM) {
            #pragma unroll 8
            for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
                *(uint4*)(&mlp_in[0]) = *(uint4*)(&mlp_intermediate[ffn_input_offset + (id - 1) * TMA_LOAD_ONCE + input_idx + i]);
                #pragma unroll
                for (int d = 0; d < NUM_PER_THREAD; d++) {
                    tmp += __half2float(mlp_in[d]) * __half2float(weight[((id - 1) % 2) * TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
                }
            }
        }
    }
    bar[(ffn_input_per_block / TMA_LOAD_ONCE - 1) % 2].wait(std::move(token[(ffn_input_per_block / TMA_LOAD_ONCE - 1) % 2]));
    if (weight_idx < HEAD_DIM) {
        #pragma unroll 8
        for (int i = 0; i < TMA_LOAD_ONCE; i += NUM_PER_ROW) {
            *(uint4*)(&mlp_in[0]) = *(uint4*)(&mlp_intermediate[ffn_input_offset + ((ffn_input_per_block / TMA_LOAD_ONCE) - 1) * TMA_LOAD_ONCE + input_idx + i]);
            #pragma unroll
            for (int d = 0; d < NUM_PER_THREAD; d++) {
                tmp += __half2float(mlp_in[d]) * __half2float(weight[TMA_LOAD_ONCE_NUM + weight_idx * TMA_LOAD_ONCE + (input_idx + i + d)]);
            }
        }
    }
    #pragma unroll
    for (int mask = (NUM_THREAD_PER_ROW >> 1); mask > 0; mask >>= 1) {
        tmp += __shfl_down_sync(0xffffffff, tmp, mask);
    }
    if (lane_id % NUM_THREAD_PER_ROW == 0 && weight_idx < HEAD_DIM) {
        local_qkv[weight_idx] = __float2half(tmp);
    }
    block.sync();

    // ==================== Cluster Reduce MLP Down ====================
    size = HEAD_DIM * sizeof(half);
    src_addr = static_cast<uint32_t>(__cvta_generic_to_shared(local_qkv));
    dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(weight));
    cluster_reduce<CLUSTER_SIZE, Stage::LINEAR>(
        size, tid, HEAD_DIM, cluster_block_id,
        src_addr, dst_addr, bar_ptr,
        neighbor_dst_bar, local_qkv, weight);
    cluster.sync();

    // ==================== Final Residual Connection ====================
    // Pythia parallel residual: output = input + attn_output + mlp_output
    if (cluster_block_id == 0) {
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            int out_idx = down_out_offset + d;
            float mlp_val = __half2float(local_qkv[d]) + __half2float(mlp_down_bias[out_idx]);
            float input_val = __half2float(input[out_idx]);
            float attn_val = __half2float(attn_output[out_idx]);
            output[out_idx] = __float2half(input_val + attn_val + mlp_val);
        }
    }
}
