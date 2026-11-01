#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

using RuntimeGetVersion = cudaError_t(CUDARTAPI*)(int*);
constexpr int kVirtualRuntimeVersion = 13000;

}  // namespace

extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* runtimeVersion) {
  const char* requestedVersion = std::getenv("CUDA_COMPAT_RUNTIME_VERSION");
  if (requestedVersion == nullptr) {
    const auto realFunction = reinterpret_cast<RuntimeGetVersion>(
        dlsym(RTLD_NEXT, "cudaRuntimeGetVersion"));
    if (realFunction == nullptr) return cudaErrorUnknown;
    return realFunction(runtimeVersion);
  }

  if (runtimeVersion == nullptr ||
      std::strcmp(requestedVersion, "13000") != 0) {
    return cudaErrorInvalidValue;
  }
  *runtimeVersion = kVirtualRuntimeVersion;
  return cudaSuccess;
}
