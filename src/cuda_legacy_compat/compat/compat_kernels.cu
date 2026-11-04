#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdint.h>

extern "C" __global__ void compat_vector_add_i32(const int32_t* left,
                                                   const int32_t* right,
                                                   int32_t* output,
                                                   uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = left[index] + right[index];
}

extern "C" __global__ void compat_vector_add_f32(const float* left,
                                                   const float* right,
                                                   float* output,
                                                   uint64_t count,
                                                   float alpha) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = left[index] + alpha * right[index];
}

extern "C" __global__ void compat_vector_add_f16(const __half* left,
                                                   const __half* right,
                                                   __half* output,
                                                   uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float sum = __half2float(left[index]) + __half2float(right[index]);
    output[index] = __float2half_rn(sum);
  }
}

extern "C" __global__ void compat_arange_i64(int64_t* output,
                                               uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = static_cast<int64_t>(index);
}

extern "C" __global__ void compat_embedding_f32(const float* weight,
                                                  const int64_t* indices,
                                                  float* output,
                                                  uint64_t token_count,
                                                  uint64_t embedding_size,
                                                  uint64_t index_size1,
                                                  uint64_t index_stride0,
                                                  uint64_t index_stride1,
                                                  uint64_t index_rank) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t output_count = token_count * embedding_size;
  if (index < output_count) {
    const uint64_t token = index / embedding_size;
    const uint64_t feature = index - token * embedding_size;
    const uint64_t offset = index_rank == 1
        ? token * index_stride0
        : (token / index_size1) * index_stride0 + (token % index_size1) * index_stride1;
    const int64_t row = indices[offset];
    output[index] = weight[static_cast<uint64_t>(row) * embedding_size + feature];
  }
}

extern "C" __global__ void compat_layer_norm_f32(
    const float* input, const float* weight, const float* bias, float* output,
    uint64_t row_count, uint64_t normalized_size, float epsilon) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = row_count * normalized_size;
  if (index >= total) return;

  const uint64_t row = index / normalized_size;
  const uint64_t offset = row * normalized_size;
  double sum = 0.0;
  for (uint64_t feature = 0; feature < normalized_size; ++feature)
    sum += static_cast<double>(input[offset + feature]);
  const double mean = sum / static_cast<double>(normalized_size);
  double squared_difference = 0.0;
  for (uint64_t feature = 0; feature < normalized_size; ++feature) {
    const double difference = static_cast<double>(input[offset + feature]) - mean;
    squared_difference += difference * difference;
  }
  const float inverse_std = rsqrtf(
      static_cast<float>(squared_difference / static_cast<double>(normalized_size)) +
      epsilon);
  float normalized = (input[index] - static_cast<float>(mean)) * inverse_std;
  normalized = normalized * weight[index - offset] + bias[index - offset];
  output[index] = normalized;
}

// Batched, tiled float32 matrix multiply (#184): output[b] = alpha * left[b] @ right[b] + beta * bias[b].
// Every operand is addressed through its own strides (64-bit), so transposes, views and a broadcast bias (stride 0)
// need no copies. A 256-thread block computes a 64x64 output tile, 4x4 values per thread, in 16-deep steps of the
// inner dimension staged through shared memory. The grid's z dimension is the batch.
#define COMPAT_GEMM_TILE_M 64
#define COMPAT_GEMM_TILE_N 64
#define COMPAT_GEMM_TILE_K 16
extern "C" __global__ void compat_matrix_multiply_f32(
    const float* left, const float* right, const float* bias, float* output, int rows, int columns, int inner,
    uint64_t left_stride0, uint64_t left_stride1, uint64_t right_stride0, uint64_t right_stride1,
    uint64_t bias_stride0, uint64_t bias_stride1, uint64_t left_batch, uint64_t right_batch, uint64_t bias_batch,
    uint64_t output_batch, float alpha, float beta) {
  __shared__ float left_tile[COMPAT_GEMM_TILE_K][COMPAT_GEMM_TILE_M + 4];
  __shared__ float right_tile[COMPAT_GEMM_TILE_K][COMPAT_GEMM_TILE_N + 4];
  const uint64_t batch = blockIdx.z;
  left += batch * left_batch;
  right += batch * right_batch;
  output += batch * output_batch;
  const int thread = threadIdx.y * 16 + threadIdx.x;
  const int row0 = blockIdx.y * COMPAT_GEMM_TILE_M;
  const int column0 = blockIdx.x * COMPAT_GEMM_TILE_N;
  float sum[4][4] = {{0.0f}};
  for (int k0 = 0; k0 < inner; k0 += COMPAT_GEMM_TILE_K) {
    for (int i = 0; i < 4; ++i) {
      const int element = thread + i * 256;
      const int r = element / COMPAT_GEMM_TILE_K, k = element % COMPAT_GEMM_TILE_K;
      const int global_row = row0 + r, global_k = k0 + k;
      left_tile[k][r] = (global_row < rows && global_k < inner)
          ? left[static_cast<uint64_t>(global_row) * left_stride0 + static_cast<uint64_t>(global_k) * left_stride1]
          : 0.0f;
      const int kr = element / COMPAT_GEMM_TILE_N, c = element % COMPAT_GEMM_TILE_N;
      const int global_kr = k0 + kr, global_column = column0 + c;
      right_tile[kr][c] = (global_kr < inner && global_column < columns)
          ? right[static_cast<uint64_t>(global_kr) * right_stride0 + static_cast<uint64_t>(global_column) * right_stride1]
          : 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < COMPAT_GEMM_TILE_K; ++k) {
      float a[4], b[4];
      for (int i = 0; i < 4; ++i) a[i] = left_tile[k][threadIdx.y * 4 + i];
      for (int j = 0; j < 4; ++j) b[j] = right_tile[k][threadIdx.x * 4 + j];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) sum[i][j] = __fmaf_rn(a[i], b[j], sum[i][j]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i) {
    const int row = row0 + threadIdx.y * 4 + i;
    if (row >= rows) continue;
    for (int j = 0; j < 4; ++j) {
      const int column = column0 + threadIdx.x * 4 + j;
      if (column >= columns) continue;
      float value = alpha * sum[i][j];
      if (bias != nullptr)
        value = __fmaf_rn(beta, bias[batch * bias_batch + static_cast<uint64_t>(row) * bias_stride0 +
                                     static_cast<uint64_t>(column) * bias_stride1], value);
      output[static_cast<uint64_t>(row) * columns + column] = value;
    }
  }
}

extern "C" __global__ void compat_relu_f32(const float* input, float* output,
                                           uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = input[index];
    output[index] = value < 0.0f ? 0.0f : value;
  }
}
