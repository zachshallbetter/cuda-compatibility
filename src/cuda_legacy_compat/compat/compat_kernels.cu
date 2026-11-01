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

extern "C" __global__ void compat_matrix_multiply_f32(const float* left,
                                                       const float* right,
                                                       float* output, int rows,
                                                       int columns, int inner,
                                                       uint64_t left_stride0,
                                                       uint64_t left_stride1,
                                                       uint64_t right_stride0,
                                                       uint64_t right_stride1) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int count = rows * columns;
  if (index >= count) return;

  const int row = index / columns;
  const int column = index - row * columns;
  float value = 0.0f;
  for (int position = 0; position < inner; ++position) {
    value = __fmaf_rn(left[static_cast<uint64_t>(row) * left_stride0 +
                           static_cast<uint64_t>(position) * left_stride1],
                      right[static_cast<uint64_t>(position) * right_stride0 +
                            static_cast<uint64_t>(column) * right_stride1],
                      value);
  }
  output[index] = value;
}

extern "C" __global__ void compat_relu_f32(const float* input, float* output,
                                           uint64_t count) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = input[index];
    output[index] = value < 0.0f ? 0.0f : value;
  }
}
