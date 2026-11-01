#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/../.." && pwd)"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-12.4}"
temporary_dir="$(mktemp -d /tmp/cuda-compat-runtime-facade.XXXXXX)"
trap 'rm -rf "$temporary_dir"' EXIT

g++ -std=c++17 -fPIC -shared -I"$cuda_root/include" \
  "$project_root/src/cuda_legacy_compat/compat/cuda_runtime_version_facade.cpp" \
  -ldl -o "$temporary_dir/libcuda_runtime_version_facade.so"
"$cuda_root/bin/nvcc" --cudart=shared -O2 -std=c++17 -arch=sm_52 \
  "$project_root/experiments/runtime-facade/runtime_probe.cu" \
  -o "$temporary_dir/runtime_probe"
"$cuda_root/bin/nvcc" --cudart=shared -O2 -std=c++17 -arch=sm_52 \
  "$project_root/experiments/maxwell/vector_add.cu" \
  -o "$temporary_dir/vector_add"

for device in 0 1 2; do
  printf 'baseline physical_device=%s ' "$device"
  CUDA_VISIBLE_DEVICES="$device" "$temporary_dir/runtime_probe"
  printf 'facade physical_device=%s ' "$device"
  CUDA_VISIBLE_DEVICES="$device" CUDA_COMPAT_RUNTIME_VERSION=13000 \
    LD_PRELOAD="$temporary_dir/libcuda_runtime_version_facade.so" \
    "$temporary_dir/runtime_probe"
  CUDA_VISIBLE_DEVICES="$device" CUDA_COMPAT_RUNTIME_VERSION=13000 \
    LD_PRELOAD="$temporary_dir/libcuda_runtime_version_facade.so" \
    "$temporary_dir/vector_add"
done

if CUDA_COMPAT_RUNTIME_VERSION=13001 \
  LD_PRELOAD="$temporary_dir/libcuda_runtime_version_facade.so" \
  "$temporary_dir/runtime_probe" >/dev/null 2>&1; then
  printf 'unsupported virtual runtime version was accepted\n' >&2
  exit 1
fi
printf 'PASS unsupported virtual runtime version rejected\n'
