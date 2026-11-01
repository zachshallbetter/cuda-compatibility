#!/usr/bin/env bash
set -euo pipefail

nvcc_bin="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
source_file="$(dirname "$0")/vector_add.cu"
binary="$(mktemp /tmp/titan-maxwell-sm52.XXXXXX)"
trap 'rm -f "$binary"' EXIT

"$nvcc_bin" -O2 -std=c++17 -arch=sm_52 "$source_file" -o "$binary"
for device in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$device" "$binary"
done
