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

// ── Elementwise, row-wise and reduction kernels (#184 part C) ─────────────────────────────────────────────────
// Up to six dimensions; `a` and `b` are the two operands' element strides for each output dimension (0 broadcasts).
struct CompatShape {
  int64_t size[6];
  int64_t a[6];
  int64_t b[6];
};

__device__ inline void compat_offsets(int64_t index, const CompatShape& shape, int ndim, int64_t* a, int64_t* b) {
  int64_t left = 0, right = 0;
  for (int d = ndim - 1; d >= 0; --d) {
    const int64_t i = index % shape.size[d];
    index /= shape.size[d];
    left += i * shape.a[d];
    right += i * shape.b[d];
  }
  *a = left;
  *b = right;
}

// op: 0 a + alpha*b, 1 a - alpha*b, 2 a*b, 3 a/b, 4 a + scalar, 5 a*scalar, 6 a/scalar, 7 scalar - a.
extern "C" __global__ void compat_binary_f32(const float* left, const float* right, float* output, int64_t count,
                                             int ndim, CompatShape shape, int op, float alpha, float scalar) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t a, b;
    compat_offsets(i, shape, ndim, &a, &b);
    const float x = left[a];
    float value;
    switch (op) {
      case 0: value = __fmaf_rn(alpha, right[b], x); break;
      case 1: value = __fmaf_rn(-alpha, right[b], x); break;
      case 2: value = x * right[b]; break;
      case 3: value = x / right[b]; break;
      case 4: value = x + scalar; break;
      case 5: value = x * scalar; break;
      case 6: value = x / scalar; break;
      default: value = scalar - x; break;
    }
    output[i] = value;
  }
}

// op: 0 GELU (erf), 1 GELU (tanh), 2 fill with `value`.
extern "C" __global__ void compat_unary_f32(const float* input, float* output, int64_t count, int op, float value) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (op == 2) {
      output[i] = value;
      continue;
    }
    const float x = input[i];
    output[i] = op == 0 ? 0.5f * x * (1.0f + erff(x * 0.70710678118654752f))
                        : 0.5f * x * (1.0f + tanhf(0.79788456080286536f * (x + 0.044715f * x * x * x)));
  }
}

extern "C" __global__ void compat_gelu_backward_f32(const float* grad, const float* input, float* output,
                                                   int64_t count, int approximate) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float x = input[i];
    float derivative;
    if (approximate == 0) {
      const float cdf = 0.5f * (1.0f + erff(x * 0.70710678118654752f));
      const float pdf = expf(-0.5f * x * x) * 0.39894228040143268f;
      derivative = cdf + x * pdf;
    } else {
      const float k = 0.79788456080286536f;
      const float inner = k * (x + 0.044715f * x * x * x);
      const float t = tanhf(inner);
      derivative = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * x * x);
    }
    output[i] = grad[i] * derivative;
  }
}

// One 256-thread block per row; returns the block-wide result to every thread.
__device__ inline float compat_block_reduce(float value, float* scratch, bool take_max) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width)
      scratch[threadIdx.x] = take_max ? fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + width])
                                      : scratch[threadIdx.x] + scratch[threadIdx.x + width];
    __syncthreads();
  }
  const float result = scratch[0];
  __syncthreads();
  return result;
}

// mode: 0 softmax, 1 log-softmax, 2 safe softmax (a row that is all -inf, fully masked, gives zeros, not NaN).
extern "C" __global__ void compat_softmax_rows_f32(const float* input, float* output, int64_t rows, int64_t columns,
                                                   int mode) {
  __shared__ float scratch[256];
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + row * columns;
  float* y = output + row * columns;
  float m = -__int_as_float(0x7f800000);
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) m = fmaxf(m, x[c]);
  m = compat_block_reduce(m, scratch, true);
  if (mode == 2 && m == -__int_as_float(0x7f800000)) {   // every thread sees the same m, so all leave together
    for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) y[c] = 0.0f;
    return;
  }
  float s = 0.0f;
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) s += expf(x[c] - m);
  s = compat_block_reduce(s, scratch, false);
  const float log_s = logf(s);
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x)
    y[c] = mode == 1 ? x[c] - m - log_s : expf(x[c] - m) / s;
}

extern "C" __global__ void compat_softmax_backward_rows_f32(const float* grad, const float* output, float* grad_input,
                                                            int64_t rows, int64_t columns, int log_softmax) {
  __shared__ float scratch[256];
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const float* g = grad + row * columns;
  const float* y = output + row * columns;
  float* gi = grad_input + row * columns;
  float dot = 0.0f;
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) dot += log_softmax ? g[c] : g[c] * y[c];
  dot = compat_block_reduce(dot, scratch, false);
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x)
    gi[c] = log_softmax ? g[c] - expf(y[c]) * dot : y[c] * (g[c] - dot);
}

// Layer normalization over the last `columns` values of each row, saving each row's mean and 1/std for backward.
extern "C" __global__ void compat_layer_norm_forward_f32(const float* input, const float* weight, const float* bias,
                                                         float* output, float* mean_out, float* rstd_out,
                                                         int64_t rows, int64_t columns, float epsilon) {
  __shared__ float scratch[256];
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + row * columns;
  float s = 0.0f;
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) s += x[c];
  const float mean = compat_block_reduce(s, scratch, false) / columns;
  float v = 0.0f;
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) v += (x[c] - mean) * (x[c] - mean);
  const float rstd = rsqrtf(compat_block_reduce(v, scratch, false) / columns + epsilon);
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) {
    float value = (x[c] - mean) * rstd;
    if (weight != nullptr) value *= weight[c];
    if (bias != nullptr) value += bias[c];
    output[row * columns + c] = value;
  }
  if (threadIdx.x == 0) {
    mean_out[row] = mean;
    rstd_out[row] = rstd;
  }
}

extern "C" __global__ void compat_layer_norm_backward_input_f32(const float* grad, const float* input, const float* mean,
                                                                const float* rstd, const float* weight,
                                                                float* grad_input, int64_t rows, int64_t columns) {
  __shared__ float scratch[256];
  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const float* g = grad + row * columns;
  const float* x = input + row * columns;
  const float m = mean[row], r = rstd[row];
  float sum_gw = 0.0f, sum_gw_xhat = 0.0f;
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) {
    const float gw = weight != nullptr ? g[c] * weight[c] : g[c];
    sum_gw += gw;
    sum_gw_xhat += gw * (x[c] - m) * r;
  }
  sum_gw = compat_block_reduce(sum_gw, scratch, false);
  sum_gw_xhat = compat_block_reduce(sum_gw_xhat, scratch, false);
  for (int64_t c = threadIdx.x; c < columns; c += blockDim.x) {
    const float gw = weight != nullptr ? g[c] * weight[c] : g[c];
    const float xhat = (x[c] - m) * r;
    grad_input[row * columns + c] = r * (gw - sum_gw / columns - xhat * sum_gw_xhat / columns);
  }
}

// One block per column: the weight and bias gradients sum over every row.
extern "C" __global__ void compat_layer_norm_backward_params_f32(const float* grad, const float* input,
                                                                 const float* mean, const float* rstd,
                                                                 float* grad_weight, float* grad_bias,
                                                                 int64_t rows, int64_t columns) {
  __shared__ float scratch[256];
  const int64_t column = blockIdx.x;
  if (column >= columns) return;
  float sw = 0.0f, sb = 0.0f;
  for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
    const float g = grad[row * columns + column];
    sw += g * (input[row * columns + column] - mean[row]) * rstd[row];
    sb += g;
  }
  sw = compat_block_reduce(sw, scratch, false);
  sb = compat_block_reduce(sb, scratch, false);
  if (threadIdx.x == 0) {
    if (grad_weight != nullptr) grad_weight[column] = sw;
    if (grad_bias != nullptr) grad_bias[column] = sb;
  }
}

// Sum over any set of dimensions: one block per output element. `kept` holds the kept dimensions' sizes (size) and
// input strides (a); `reduced` the reduced ones'.
extern "C" __global__ void compat_reduce_sum_f32(const float* input, float* output, int64_t outputs, int kept_ndim,
                                                 CompatShape kept, int reduced_ndim, CompatShape reduced,
                                                 int64_t reduced_count) {
  __shared__ float scratch[256];
  const int64_t o = blockIdx.x;
  if (o >= outputs) return;
  int64_t base, unused;
  compat_offsets(o, kept, kept_ndim, &base, &unused);
  float s = 0.0f;
  for (int64_t r = threadIdx.x; r < reduced_count; r += blockDim.x) {
    int64_t offset;
    compat_offsets(r, reduced, reduced_ndim, &offset, &unused);
    s += input[base + offset];
  }
  s = compat_block_reduce(s, scratch, false);
  if (threadIdx.x == 0) output[o] = s;
}

// ── Fewer copies (#184 part B) ────────────────────────────────────────────────────────────────────────────────
// Any 1-, 2-, 4- or 8-byte element type, from strides `a` (input) to strides `b` (output): clone and contiguous.
extern "C" __global__ void compat_strided_copy(const char* input, char* output, int64_t count, int ndim,
                                               CompatShape shape, int element_size) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t a, b;
    compat_offsets(i, shape, ndim, &a, &b);
    switch (element_size) {
      case 1: output[b] = input[a]; break;
      case 2: reinterpret_cast<int16_t*>(output)[b] = reinterpret_cast<const int16_t*>(input)[a]; break;
      case 4: reinterpret_cast<int32_t*>(output)[b] = reinterpret_cast<const int32_t*>(input)[a]; break;
      default: reinterpret_cast<int64_t*>(output)[b] = reinterpret_cast<const int64_t*>(input)[a]; break;
    }
  }
}

// The embedding's weight gradient: each looked-up row adds its gradient row (grad_weight starts at zero). Rows
// outside the table or equal to padding_idx add nothing.
extern "C" __global__ void compat_embedding_backward_f32(const float* grad, const int64_t* indices, float* grad_weight,
                                                         int64_t tokens, int64_t columns, int64_t rows,
                                                         int64_t padding_idx) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < tokens * columns;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t token = i / columns;
    const int64_t row = indices[token];
    if (row == padding_idx || row < 0 || row >= rows) continue;
    atomicAdd(grad_weight + row * columns + (i - token * columns), grad[i]);
  }
}

// Negative log likelihood in one 256-thread block. reduction: 0 none (a loss per row), 1 mean, 2 sum.
// total_weight is the summed weight of the rows not ignored (0 for a batch with no reduction, as PyTorch's CPU).
extern "C" __global__ void compat_nll_loss_forward_f32(const float* input, const int64_t* target, const float* weight,
                                                       float* output, float* total_weight, int64_t rows,
                                                       int64_t columns, int64_t ignore_index, int reduction,
                                                       int batched) {
  __shared__ float scratch[256];
  float s = 0.0f, w = 0.0f;
  for (int64_t n = threadIdx.x; n < rows; n += blockDim.x) {
    const int64_t t = target[n];
    float loss = 0.0f;
    if (t != ignore_index) {
      const float wt = weight != nullptr ? weight[t] : 1.0f;
      loss = -input[n * columns + t] * wt;
      w += wt;
    }
    if (reduction == 0) output[n] = loss;
    s += loss;
  }
  s = compat_block_reduce(s, scratch, false);
  w = compat_block_reduce(w, scratch, false);
  if (threadIdx.x == 0) {
    if (reduction != 0) output[0] = reduction == 1 ? s / w : s;
    total_weight[0] = reduction == 0 && batched ? 0.0f : w;
  }
}

// grad_input starts at zero; each row not ignored gets -weight * grad at its target (divided by the total weight
// for a mean).
extern "C" __global__ void compat_nll_loss_backward_f32(const float* grad_output, const int64_t* target,
                                                        const float* weight, const float* total_weight,
                                                        float* grad_input, int64_t rows, int64_t columns,
                                                        int64_t ignore_index, int reduction) {
  for (int64_t n = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; n < rows;
       n += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = target[n];
    if (t == ignore_index) continue;
    float g = reduction == 0 ? grad_output[n] : grad_output[0];
    if (reduction == 1) g /= total_weight[0];
    grad_input[n * columns + t] = -(weight != nullptr ? weight[t] : 1.0f) * g;
  }
}

// ── Optimizer steps (#184) ──────────────────────────────────────────────────────────────────────────────────
// Elementwise over contiguous float32, as PyTorch's CPU kernels compute them: op 0 x*s, 1 lerp from x toward y by s,
// 2 x + s*y*z, 3 sqrt(x), 4 x/s, 5 x+s, 6 x + s*y/z. `out` may be x (the in-place forms).
extern "C" __global__ void compat_pointwise_f32(const float* x, const float* y, const float* z, float* out,
                                                int64_t count, int op, float s) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float a = x[i];
    float value;
    switch (op) {
      case 0: value = __fmul_rn(a, s); break;
      case 1: {
        const float d = __fsub_rn(y[i], a);
        value = fabsf(s) < 0.5f ? __fmaf_rn(s, d, a) : __fmaf_rn(-d, __fsub_rn(1.0f, s), y[i]);
        break;
      }
      case 2: value = __fadd_rn(a, __fmul_rn(__fmul_rn(s, y[i]), z[i])); break;
      case 3: value = __fsqrt_rn(a); break;
      case 4: value = __fdiv_rn(a, s); break;
      case 5: value = __fadd_rn(a, s); break;
      case 6: value = __fadd_rn(a, __fdiv_rn(__fmul_rn(s, y[i]), z[i])); break;
      default: value = y[i] > s ? a : 0.0f; break;  // threshold_backward: x = grad, y = input
    }
    out[i] = value;
  }
}

// ── Max pooling (#184) ──────────────────────────────────────────────────────────────────────────────────────
// One thread per output element, as PyTorch's CPU kernel computes it: the window starts on the first row and column
// inside the input (stepping by the dilation past the padding), the first maximum wins a tie, NaN wins any window, and
// the index is the position in its (H, W) plane.
extern "C" __global__ void compat_max_pool2d_f32(const float* input, float* output, int64_t* indices, int64_t planes,
                                                 int64_t height, int64_t width, int64_t out_height, int64_t out_width,
                                                 int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {
  const int64_t count = planes * out_height * out_width;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t ow = i % out_width, oh = (i / out_width) % out_height, plane = i / (out_width * out_height);
    int64_t hstart = oh * sh - ph, wstart = ow * sw - pw;
    const int64_t hend = min(hstart + static_cast<int64_t>(kh - 1) * dh + 1, height);
    const int64_t wend = min(wstart + static_cast<int64_t>(kw - 1) * dw + 1, width);
    while (hstart < 0) hstart += dh;
    while (wstart < 0) wstart += dw;
    const float* x = input + plane * height * width;
    int64_t best = hstart * width + wstart;
    float value = -__int_as_float(0x7f800000);
    for (int64_t h = hstart; h < hend; h += dh)
      for (int64_t w = wstart; w < wend; w += dw) {
        const float v = x[h * width + w];
        if (v > value || isnan(v)) {
          value = v;
          best = h * width + w;
        }
      }
    output[i] = value;
    indices[i] = best;
  }
}

// grad_input starts at zero; each output's gradient goes to the input position its index names (windows overlap, so
// the sums use atomics).
extern "C" __global__ void compat_max_pool2d_backward_f32(const float* grad_output, const int64_t* indices,
                                                          float* grad_input, int64_t planes, int64_t plane_size,
                                                          int64_t out_plane_size) {
  const int64_t count = planes * out_plane_size;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t index = indices[i];
    if (index >= 0 && index < plane_size)
      atomicAdd(grad_input + (i / out_plane_size) * plane_size + index, grad_output[i]);
  }
}

// ── Fused Adam and AdamW (#246) ─────────────────────────────────────────────────────────────────────────────
// One launch per parameter for the whole step, as PyTorch's fused CUDA kernel computes it: weight decay (decoupled
// for AdamW, added to the gradient for Adam), both moments, bias correction from the step read on the device, and
// the update. amsgrad and gradient scaling stay on the CPU route.
extern "C" __global__ void compat_adam_f32(float* param, const float* grad, float* exp_avg, float* exp_avg_sq,
                                           const float* step, int64_t count, float lr, float beta1, float beta2,
                                           float weight_decay, float eps, int adamw, int maximize) {
  const float t = step[0];
  const float bias_correction1 = 1.0f - powf(beta1, t);
  const float bias_correction2_sqrt = sqrtf(1.0f - powf(beta2, t));
  const float step_size = lr / bias_correction1;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float p = param[i];
    float g = maximize ? -grad[i] : grad[i];
    if (weight_decay != 0.0f) {
      if (adamw) p -= lr * weight_decay * p;
      else g += weight_decay * p;
    }
    const float m = beta1 * exp_avg[i] + (1.0f - beta1) * g;
    const float v = beta2 * exp_avg_sq[i] + (1.0f - beta2) * g * g;
    exp_avg[i] = m;
    exp_avg_sq[i] = v;
    param[i] = p - step_size * m / (sqrtf(v) / bias_correction2_sqrt + eps);
  }
}
