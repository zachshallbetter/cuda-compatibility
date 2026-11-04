#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>

namespace {

struct CachedKernels {
  std::string ptx_path;
  CUmodule module;
  CUfunction vector_add_i32;
  CUfunction vector_add_f32;
  CUfunction vector_add_f16;
  CUfunction arange_i64;
  CUfunction embedding_f32;
  CUfunction matrix_multiply_f32;
  CUfunction relu_f32;
  CUfunction layer_norm_f32;
};

std::mutex cache_mutex;
std::unordered_map<CUcontext, CachedKernels> kernels_by_context;

CUresult find_or_load_kernels(CUcontext context, const char* ptx_path,
                              CUfunction* vector_add_i32,
                              CUfunction* vector_add_f32,
                              CUfunction* arange_i64,
                              CUfunction* embedding_f32,
                              CUfunction* matrix_multiply_f32,
                              CUfunction* relu_f32,
                              CUfunction* layer_norm_f32,
                              CUfunction* vector_add_f16 = nullptr) {
  std::lock_guard<std::mutex> guard(cache_mutex);
  auto found = kernels_by_context.find(context);
  if (found != kernels_by_context.end()) {
    if (found->second.ptx_path != ptx_path) return CUDA_ERROR_INVALID_VALUE;
    *vector_add_i32 = found->second.vector_add_i32;
    *vector_add_f32 = found->second.vector_add_f32;
    if (vector_add_f16 != nullptr)
      *vector_add_f16 = found->second.vector_add_f16;
    *arange_i64 = found->second.arange_i64;
    *embedding_f32 = found->second.embedding_f32;
    *matrix_multiply_f32 = found->second.matrix_multiply_f32;
    *relu_f32 = found->second.relu_f32;
    *layer_norm_f32 = found->second.layer_norm_f32;
    return CUDA_SUCCESS;
  }

  CUmodule module = nullptr;
  CUresult result = cuModuleLoad(&module, ptx_path);
  if (result != CUDA_SUCCESS) return result;

  CachedKernels loaded{};
  loaded.module = module;
  result = cuModuleGetFunction(&loaded.vector_add_i32, module,
                               "compat_vector_add_i32");
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&loaded.vector_add_f32, module,
                                 "compat_vector_add_f32");
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&loaded.vector_add_f16, module,
                                 "compat_vector_add_f16");
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&loaded.arange_i64, module, "compat_arange_i64");
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&loaded.embedding_f32, module, "compat_embedding_f32");
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&loaded.matrix_multiply_f32, module,
                                 "compat_matrix_multiply_f32");
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&loaded.relu_f32, module, "compat_relu_f32");
  if (result == CUDA_SUCCESS)
    result = cuModuleGetFunction(&loaded.layer_norm_f32, module, "compat_layer_norm_f32");
  if (result != CUDA_SUCCESS) {
    cuModuleUnload(module);
    return result;
  }

  try {
    loaded.ptx_path = ptx_path;
    auto inserted = kernels_by_context.emplace(context, std::move(loaded));
    if (!inserted.second) {
      cuModuleUnload(module);
      if (inserted.first->second.ptx_path != ptx_path)
        return CUDA_ERROR_INVALID_VALUE;
      *vector_add_i32 = inserted.first->second.vector_add_i32;
      *vector_add_f32 = inserted.first->second.vector_add_f32;
      if (vector_add_f16 != nullptr)
        *vector_add_f16 = inserted.first->second.vector_add_f16;
      *arange_i64 = inserted.first->second.arange_i64;
      *embedding_f32 = inserted.first->second.embedding_f32;
      *matrix_multiply_f32 = inserted.first->second.matrix_multiply_f32;
      *relu_f32 = inserted.first->second.relu_f32;
      *layer_norm_f32 = inserted.first->second.layer_norm_f32;
      return CUDA_SUCCESS;
    }
    *vector_add_i32 = inserted.first->second.vector_add_i32;
    *vector_add_f32 = inserted.first->second.vector_add_f32;
    if (vector_add_f16 != nullptr)
      *vector_add_f16 = inserted.first->second.vector_add_f16;
    *arange_i64 = inserted.first->second.arange_i64;
    *embedding_f32 = inserted.first->second.embedding_f32;
    *matrix_multiply_f32 = inserted.first->second.matrix_multiply_f32;
    *relu_f32 = inserted.first->second.relu_f32;
    *layer_norm_f32 = inserted.first->second.layer_norm_f32;
  } catch (const std::bad_alloc&) {
    cuModuleUnload(module);
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    cuModuleUnload(module);
    return CUDA_ERROR_UNKNOWN;
  }
  return CUDA_SUCCESS;
}

CUresult current_kernels(const char* ptx_path, CUfunction* vector_add_i32,
                         CUfunction* vector_add_f32,
                         CUfunction* arange_i64,
                         CUfunction* embedding_f32,
                         CUfunction* matrix_multiply_f32,
                         CUfunction* relu_f32,
                         CUfunction* layer_norm_f32,
                         CUfunction* vector_add_f16 = nullptr) {
  if (ptx_path == nullptr || vector_add_i32 == nullptr || vector_add_f32 == nullptr ||
      arange_i64 == nullptr ||
      embedding_f32 == nullptr ||
      matrix_multiply_f32 == nullptr || relu_f32 == nullptr ||
      layer_norm_f32 == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) return result;
  if (context == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
  try {
    return find_or_load_kernels(context, ptx_path, vector_add_i32, vector_add_f32,
                                arange_i64,
                                embedding_f32,
                                matrix_multiply_f32, relu_f32, layer_norm_f32,
                                vector_add_f16);
  } catch (const std::bad_alloc&) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}

// A thread PyTorch never ran a CUDA call on (autograd's backward thread before its first allocation) has no current
// context; make the device's primary context (the one PyTorch uses) current there.
CUresult ensure_context(int device) {
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS || context != nullptr) return result;
  CUdevice handle = 0;
  result = cuDeviceGet(&handle, device);
  if (result != CUDA_SUCCESS) return result;
  result = cuDevicePrimaryCtxRetain(&context, handle);
  if (result != CUDA_SUCCESS) return result;
  return cuCtxSetCurrent(context);
}

// Kernels looked up by name (#184): the module current_kernels loaded for a context, then each function once.
std::mutex named_mutex;
std::unordered_map<std::string, CUfunction> named_functions;

CUresult named_kernel(const char* ptx_path, const char* name, CUfunction* function) {
  CUfunction unused[7] = {};
  CUresult result = current_kernels(ptx_path, &unused[0], &unused[1], &unused[2], &unused[3], &unused[4],
                                    &unused[5], &unused[6]);
  if (result != CUDA_SUCCESS) return result;
  CUcontext context = nullptr;
  result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) return result;
  CUmodule module = nullptr;
  {
    std::lock_guard<std::mutex> guard(cache_mutex);
    auto found = kernels_by_context.find(context);
    if (found == kernels_by_context.end()) return CUDA_ERROR_INVALID_CONTEXT;
    module = found->second.module;
  }
  try {
    const std::string key = std::to_string(reinterpret_cast<std::uintptr_t>(context)) + "/" + name;
    std::lock_guard<std::mutex> guard(named_mutex);
    auto found = named_functions.find(key);
    if (found != named_functions.end()) {
      *function = found->second;
      return CUDA_SUCCESS;
    }
    result = cuModuleGetFunction(function, module, name);
    if (result == CUDA_SUCCESS) named_functions.emplace(key, *function);
    return result;
  } catch (...) {
    return CUDA_ERROR_UNKNOWN;
  }
}

}  // namespace

// Make the device's primary context current on this thread when none is (#188). Every native launch calls it first:
// on autograd's backward thread a cached allocation makes no CUDA call, so nothing else would.
extern "C" int cuda_compat_ensure_context(int device) { return ensure_context(device); }

// Launch a kernel of the adapter's module by name with the caller's argument pointers (cuLaunchKernel's own form),
// so a new kernel needs no launcher of its own here (#184).
extern "C" int cuda_compat_launch_named(const char* ptx_path, const char* name, int device, unsigned int grid_x,
                                        unsigned int grid_y, unsigned int grid_z, unsigned int block_x,
                                        unsigned int block_y, unsigned int block_z, unsigned int shared_bytes,
                                        std::uintptr_t stream_pointer, void** arguments) {
  if (ptx_path == nullptr || name == nullptr || arguments == nullptr || grid_x == 0 || grid_y == 0 ||
      grid_z == 0 || block_x * block_y * block_z == 0 || block_x * block_y * block_z > 1024)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult result = ensure_context(device);
  if (result != CUDA_SUCCESS) return result;
  CUfunction function = nullptr;
  result = named_kernel(ptx_path, name, &function);
  if (result != CUDA_SUCCESS) return result;
  return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared_bytes,
                        reinterpret_cast<CUstream>(stream_pointer), arguments, nullptr);
}

extern "C" int cuda_compat_launch_vector_add(
    const char* ptx_path, std::uint64_t left_pointer,
    std::uint64_t right_pointer, std::uint64_t output_pointer,
    std::uint64_t count, std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || count > 2147483647ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (count == 0) return CUDA_SUCCESS;
  if (left_pointer == 0 || right_pointer == 0 || output_pointer == 0)
    return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(ptx_path, &vector_add_i32, &vector_add_f32,
                                    &arange_i64,
                                    &embedding_f32,
                                    &matrix_multiply_f32, &relu_f32, &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr left = static_cast<CUdeviceptr>(left_pointer);
  CUdeviceptr right = static_cast<CUdeviceptr>(right_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  std::uint64_t kernel_count = count;
  void* arguments[] = {&left, &right, &output, &kernel_count};
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((count + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(vector_add_i32, blocks, 1, 1, threads, 1, 1,
                        0, stream, arguments, nullptr);
}

extern "C" int cuda_compat_launch_vector_add_scaled_f32(
    const char* ptx_path, std::uint64_t left_pointer,
    std::uint64_t right_pointer, std::uint64_t output_pointer,
    std::uint64_t count, float alpha, std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || count > 1048576ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (count == 0) return CUDA_SUCCESS;
  if (left_pointer == 0 || right_pointer == 0 || output_pointer == 0)
    return CUDA_ERROR_INVALID_VALUE;
  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(ptx_path, &vector_add_i32, &vector_add_f32,
                                    &arange_i64, &embedding_f32,
                                    &matrix_multiply_f32, &relu_f32,
                                    &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;
  CUdeviceptr left = static_cast<CUdeviceptr>(left_pointer);
  CUdeviceptr right = static_cast<CUdeviceptr>(right_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  void* arguments[] = {&left, &right, &output, &count, &alpha};
  constexpr unsigned int threads = 256;
  const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(vector_add_f32, blocks, 1, 1, threads, 1, 1, 0,
                        stream, arguments, nullptr);
}

extern "C" int cuda_compat_launch_vector_add_f32(
    const char* ptx_path, std::uint64_t left_pointer,
    std::uint64_t right_pointer, std::uint64_t output_pointer,
    std::uint64_t count, std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || count > 1048576ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (count == 0) return CUDA_SUCCESS;
  if (left_pointer == 0 || right_pointer == 0 || output_pointer == 0)
    return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(ptx_path, &vector_add_i32, &vector_add_f32,
                                    &arange_i64, &embedding_f32,
                                    &matrix_multiply_f32, &relu_f32, &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr left = static_cast<CUdeviceptr>(left_pointer);
  CUdeviceptr right = static_cast<CUdeviceptr>(right_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  std::uint64_t kernel_count = count;
  float alpha = 1.0f;
  void* arguments[] = {&left, &right, &output, &kernel_count, &alpha};
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((count + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(vector_add_f32, blocks, 1, 1, threads, 1, 1, 0, stream,
                        arguments, nullptr);
}

extern "C" int cuda_compat_launch_vector_add_f16(
    const char* ptx_path, std::uint64_t left_pointer,
    std::uint64_t right_pointer, std::uint64_t output_pointer,
    std::uint64_t count, std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || count > 1048576ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (count == 0) return CUDA_SUCCESS;
  if (left_pointer == 0 || right_pointer == 0 || output_pointer == 0)
    return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUfunction vector_add_f16 = nullptr;
  CUresult result = current_kernels(
      ptx_path, &vector_add_i32, &vector_add_f32, &arange_i64, &embedding_f32,
      &matrix_multiply_f32, &relu_f32, &layer_norm_f32, &vector_add_f16);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr left = static_cast<CUdeviceptr>(left_pointer);
  CUdeviceptr right = static_cast<CUdeviceptr>(right_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  std::uint64_t kernel_count = count;
  void* arguments[] = {&left, &right, &output, &kernel_count};
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((count + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(vector_add_f16, blocks, 1, 1, threads, 1, 1,
                        0, stream, arguments, nullptr);
}

extern "C" int cuda_compat_launch_arange_i64(
    const char* ptx_path, std::uint64_t output_pointer, std::uint64_t count,
    std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || count > 1048576ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (count == 0) return CUDA_SUCCESS;
  if (output_pointer == 0) return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(ptx_path, &vector_add_i32, &vector_add_f32,
                                    &arange_i64,
                                    &embedding_f32,
                                    &matrix_multiply_f32, &relu_f32, &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  std::uint64_t kernel_count = count;
  void* arguments[] = {&output, &kernel_count};
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((count + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(arange_i64, blocks, 1, 1, threads, 1, 1, 0, stream,
                        arguments, nullptr);
}

extern "C" int cuda_compat_copy_device_to_host(
    std::uint64_t device_pointer, std::uintptr_t host_pointer,
    std::size_t byte_count) {
  if (device_pointer == 0 || host_pointer == 0 || byte_count == 0)
    return CUDA_ERROR_INVALID_VALUE;
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) return result;
  if (context == nullptr) return CUDA_ERROR_INVALID_CONTEXT;
  return cuMemcpyDtoH(reinterpret_cast<void*>(host_pointer),
                      static_cast<CUdeviceptr>(device_pointer), byte_count);
}

extern "C" int cuda_compat_launch_embedding_f32(
    const char* ptx_path, std::uint64_t weight_pointer,
    std::uint64_t indices_pointer, std::uint64_t output_pointer,
    std::uint64_t token_count, std::uint64_t embedding_size,
    std::uint64_t index_size1, std::uint64_t index_stride0,
    std::uint64_t index_stride1, std::uint64_t index_rank,
    std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || token_count > 1048576ULL || embedding_size > 4096ULL ||
      token_count * embedding_size > 1048576ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (token_count == 0 || embedding_size == 0) return CUDA_SUCCESS;
  if (weight_pointer == 0 || indices_pointer == 0 || output_pointer == 0)
    return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(ptx_path, &vector_add_i32, &vector_add_f32,
                                    &arange_i64,
                                    &embedding_f32, &matrix_multiply_f32,
                                    &relu_f32, &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr weight = static_cast<CUdeviceptr>(weight_pointer);
  CUdeviceptr indices = static_cast<CUdeviceptr>(indices_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  std::uint64_t kernel_token_count = token_count;
  std::uint64_t kernel_embedding_size = embedding_size;
  std::uint64_t kernel_index_size1 = index_size1;
  std::uint64_t kernel_index_stride0 = index_stride0;
  std::uint64_t kernel_index_stride1 = index_stride1;
  std::uint64_t kernel_index_rank = index_rank;
  void* arguments[] = {&weight, &indices, &output, &kernel_token_count,
                       &kernel_embedding_size, &kernel_index_size1,
                       &kernel_index_stride0, &kernel_index_stride1,
                       &kernel_index_rank};
  constexpr unsigned int threads = 256;
  const std::uint64_t output_count = token_count * embedding_size;
  const unsigned int blocks = static_cast<unsigned int>(
      (output_count + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(embedding_f32, blocks, 1, 1, threads, 1, 1, 0, stream,
                        arguments, nullptr);
}

extern "C" int cuda_compat_launch_matrix_multiply_f32(
    const char* ptx_path, std::uint64_t left_pointer, std::uint64_t right_pointer, std::uint64_t bias_pointer,
    std::uint64_t output_pointer, std::uint64_t batch, std::uint64_t rows, std::uint64_t columns,
    std::uint64_t inner, std::uint64_t left_stride0, std::uint64_t left_stride1, std::uint64_t right_stride0,
    std::uint64_t right_stride1, std::uint64_t bias_stride0, std::uint64_t bias_stride1, std::uint64_t left_batch,
    std::uint64_t right_batch, std::uint64_t bias_batch, std::uint64_t output_batch, float alpha, float beta,
    std::uintptr_t stream_pointer) {
  // The grid holds 65535 tiles of 64 per dimension and a batch of 65535; the kernel indexes in 64 bits.
  if (ptx_path == nullptr || rows > 2147483647ULL || columns > 2147483647ULL || inner > 2147483647ULL ||
      (rows + 63) / 64 > 65535ULL || (columns + 63) / 64 > 2147483647ULL || batch > 65535ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (rows == 0 || columns == 0 || batch == 0) return CUDA_SUCCESS;
  if (output_pointer == 0 || (inner > 0 && (left_pointer == 0 || right_pointer == 0)))
    return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(ptx_path, &vector_add_i32, &vector_add_f32,
                                    &arange_i64,
                                    &embedding_f32,
                                    &matrix_multiply_f32, &relu_f32, &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr left = static_cast<CUdeviceptr>(left_pointer);
  CUdeviceptr right = static_cast<CUdeviceptr>(right_pointer);
  CUdeviceptr bias = static_cast<CUdeviceptr>(bias_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  int kernel_rows = static_cast<int>(rows);
  int kernel_columns = static_cast<int>(columns);
  int kernel_inner = static_cast<int>(inner);
  std::uint64_t strides[10] = {left_stride0, left_stride1, right_stride0, right_stride1, bias_stride0,
                               bias_stride1, left_batch, right_batch, bias_batch, output_batch};
  float kernel_alpha = alpha;
  float kernel_beta = beta;
  void* arguments[] = {&left, &right, &bias, &output, &kernel_rows, &kernel_columns, &kernel_inner,
                       &strides[0], &strides[1], &strides[2], &strides[3], &strides[4], &strides[5],
                       &strides[6], &strides[7], &strides[8], &strides[9], &kernel_alpha, &kernel_beta};
  const unsigned int grid_x = static_cast<unsigned int>((columns + 63) / 64);
  const unsigned int grid_y = static_cast<unsigned int>((rows + 63) / 64);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(matrix_multiply_f32, grid_x, grid_y, static_cast<unsigned int>(batch), 16, 16, 1, 0,
                        stream, arguments, nullptr);
}

extern "C" int cuda_compat_launch_relu_f32(
    const char* ptx_path, std::uint64_t input_pointer,
    std::uint64_t output_pointer, std::uint64_t count,
    std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || count > 1048576ULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (count == 0) return CUDA_SUCCESS;
  if (input_pointer == 0 || output_pointer == 0)
    return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(ptx_path, &vector_add_i32, &vector_add_f32,
                                    &arange_i64,
                                    &embedding_f32,
                                    &matrix_multiply_f32, &relu_f32, &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr input = static_cast<CUdeviceptr>(input_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  std::uint64_t kernel_count = count;
  void* arguments[] = {&input, &output, &kernel_count};
  constexpr unsigned int threads = 256;
  const unsigned int blocks =
      static_cast<unsigned int>((count + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(relu_f32, blocks, 1, 1, threads, 1, 1, 0,
                        stream, arguments, nullptr);
}

extern "C" int cuda_compat_launch_layer_norm_f32(
    const char* ptx_path, std::uint64_t input_pointer,
    std::uint64_t weight_pointer, std::uint64_t bias_pointer,
    std::uint64_t output_pointer, std::uint64_t row_count,
    std::uint64_t normalized_size, float epsilon,
    std::uintptr_t stream_pointer) {
  if (ptx_path == nullptr || normalized_size == 0 || normalized_size > 4096ULL ||
      row_count * normalized_size > 1048576ULL || epsilon <= 0.0f)
    return CUDA_ERROR_INVALID_VALUE;
  if (row_count == 0) return CUDA_SUCCESS;
  if (input_pointer == 0 || weight_pointer == 0 || bias_pointer == 0 ||
      output_pointer == 0)
    return CUDA_ERROR_INVALID_VALUE;

  CUfunction vector_add_i32 = nullptr;
  CUfunction vector_add_f32 = nullptr;
  CUfunction arange_i64 = nullptr;
  CUfunction embedding_f32 = nullptr;
  CUfunction matrix_multiply_f32 = nullptr;
  CUfunction relu_f32 = nullptr;
  CUfunction layer_norm_f32 = nullptr;
  CUresult result = current_kernels(
      ptx_path, &vector_add_i32, &vector_add_f32, &arange_i64, &embedding_f32,
      &matrix_multiply_f32, &relu_f32, &layer_norm_f32);
  if (result != CUDA_SUCCESS) return result;

  CUdeviceptr input = static_cast<CUdeviceptr>(input_pointer);
  CUdeviceptr weight = static_cast<CUdeviceptr>(weight_pointer);
  CUdeviceptr bias = static_cast<CUdeviceptr>(bias_pointer);
  CUdeviceptr output = static_cast<CUdeviceptr>(output_pointer);
  std::uint64_t kernel_rows = row_count;
  std::uint64_t kernel_normalized_size = normalized_size;
  void* arguments[] = {&input, &weight, &bias, &output, &kernel_rows,
                       &kernel_normalized_size, &epsilon};
  constexpr unsigned int threads = 128;
  const unsigned int blocks = static_cast<unsigned int>(
      (row_count * normalized_size + threads - 1) / threads);
  auto stream = reinterpret_cast<CUstream>(stream_pointer);
  return cuLaunchKernel(layer_norm_f32, blocks, 1, 1, threads, 1, 1, 0, stream,
                        arguments, nullptr);
}

extern "C" std::size_t cuda_compat_cached_context_count() {
  try {
    std::lock_guard<std::mutex> guard(cache_mutex);
    return kernels_by_context.size();
  } catch (...) {
    return 0;
  }
}
