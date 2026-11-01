#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

__global__ void vector_add(const int* a, const int* b, int* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = a[i] + b[i];
}

int main() {
  constexpr int count = 1 << 20;
  constexpr int threads = 256;
  std::vector<int> a(count), b(count), out(count);
  for (int i = 0; i < count; ++i) {
    a[i] = i;
    b[i] = count - i;
  }

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "no CUDA device: %s\n", cudaGetErrorString(status));
    return 2;
  }
  int device = 0;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "cudaGetDevice: %s\n", cudaGetErrorString(status));
    return 2;
  }

  int* device_a = nullptr;
  int* device_b = nullptr;
  int* device_out = nullptr;
  const size_t bytes = count * sizeof(int);
  if ((status = cudaMalloc(&device_a, bytes)) != cudaSuccess ||
      (status = cudaMalloc(&device_b, bytes)) != cudaSuccess ||
      (status = cudaMalloc(&device_out, bytes)) != cudaSuccess ||
      (status = cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice)) != cudaSuccess ||
      (status = cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice)) != cudaSuccess) {
    std::fprintf(stderr, "setup failed: %s\n", cudaGetErrorString(status));
    return 2;
  }

  vector_add<<<(count + threads - 1) / threads, threads>>>(device_a, device_b, device_out, count);
  status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status == cudaSuccess)
    status = cudaMemcpy(out.data(), device_out, bytes, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "kernel or copy failed: %s\n", cudaGetErrorString(status));
    return 2;
  }

  for (int i = 0; i < count; ++i) {
    if (out[i] != count) {
      std::fprintf(stderr, "mismatch at %d: got %d expected %d\n", i, out[i], count);
      return 1;
    }
  }
  std::printf("PASS device=%d elements=%d result=%d\n", device, count, out[0]);
  cudaFree(device_out);
  cudaFree(device_b);
  cudaFree(device_a);
  return 0;
}
