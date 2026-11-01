#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main() {
  int runtime_version = 0;
  int driver_version = 0;
  cudaError_t status = cudaRuntimeGetVersion(&runtime_version);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n", cudaGetErrorString(status));
    return 2;
  }
  status = cudaDriverGetVersion(&driver_version);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaDriverGetVersion failed: %s\n", cudaGetErrorString(status));
    return 2;
  }

  int device_count = 0;
  status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count < 1) {
    std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(status));
    return 2;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, 0);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(status));
    return 2;
  }

  const char* requested_version = std::getenv("CUDA_COMPAT_RUNTIME_VERSION");
  if (requested_version != nullptr &&
      (runtime_version != 13000 || properties.major != 5 || properties.minor != 2)) {
    std::fprintf(stderr, "unexpected facade result: runtime=%d compute=%d.%d\n",
                 runtime_version, properties.major, properties.minor);
    return 1;
  }

  std::printf("runtime=%d driver=%d device=%s compute=%d.%d\n", runtime_version,
              driver_version, properties.name, properties.major, properties.minor);
  return 0;
}
