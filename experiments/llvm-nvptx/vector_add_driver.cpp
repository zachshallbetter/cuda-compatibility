#include <cuda.h>

#include <cstdio>
#include <vector>

static bool check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return true;
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::fprintf(stderr, "%s failed: %s: %s\n", operation,
               name ? name : "unknown", description ? description : "no description");
  return false;
}

#define CHECK_CUDA(call) \
  do {                   \
    if (!check((call), #call)) return 2; \
  } while (false)

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <PTX-or-cubin>\n", argv[0]);
    return 2;
  }

  constexpr int count = 1 << 20;
  constexpr int threads = 256;
  const size_t bytes = static_cast<size_t>(count) * sizeof(int);
  std::vector<int> a(count), b(count), out(count);
  for (int i = 0; i < count; ++i) {
    a[i] = i;
    b[i] = count - i;
  }

  CHECK_CUDA(cuInit(0));
  int device_count = 0;
  CHECK_CUDA(cuDeviceGetCount(&device_count));
  if (device_count < 1) {
    std::fprintf(stderr, "no CUDA device visible\n");
    return 2;
  }

  CUdevice device;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUfunction kernel = nullptr;
  CUdeviceptr device_a = 0;
  CUdeviceptr device_b = 0;
  CUdeviceptr device_out = 0;
  CHECK_CUDA(cuDeviceGet(&device, 0));
  CHECK_CUDA(cuCtxCreate(&context, 0, device));
  CHECK_CUDA(cuModuleLoad(&module, argv[1]));
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, "_Z10vector_addPKiS0_Pii"));
  CHECK_CUDA(cuMemAlloc(&device_a, bytes));
  CHECK_CUDA(cuMemAlloc(&device_b, bytes));
  CHECK_CUDA(cuMemAlloc(&device_out, bytes));
  CHECK_CUDA(cuMemcpyHtoD(device_a, a.data(), bytes));
  CHECK_CUDA(cuMemcpyHtoD(device_b, b.data(), bytes));

  int kernel_count = count;
  void* arguments[] = {&device_a, &device_b, &device_out, &kernel_count};
  const unsigned int blocks = (count + threads - 1) / threads;
  CHECK_CUDA(cuLaunchKernel(kernel, blocks, 1, 1, threads, 1, 1, 0, nullptr,
                            arguments, nullptr));
  CHECK_CUDA(cuCtxSynchronize());
  CHECK_CUDA(cuMemcpyDtoH(out.data(), device_out, bytes));

  for (int i = 0; i < count; ++i) {
    if (out[i] != count) {
      std::fprintf(stderr, "mismatch at %d: got %d expected %d\n", i, out[i], count);
      return 1;
    }
  }

  std::printf("PASS device=0 elements=%d result=%d module=%s\n", count, out[0], argv[1]);
  CHECK_CUDA(cuMemFree(device_out));
  CHECK_CUDA(cuMemFree(device_b));
  CHECK_CUDA(cuMemFree(device_a));
  CHECK_CUDA(cuModuleUnload(module));
  CHECK_CUDA(cuCtxDestroy(context));
  return 0;
}
