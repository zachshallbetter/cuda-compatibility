#!/usr/bin/env bash
set -euo pipefail

llvm_root="${LLVM_ROOT:-/tmp/cuda-compat-llvm-23.1.4/LLVM-23.1.4-Linux-X64}"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-12.4}"
source_dir="$(cd "$(dirname "$0")/../maxwell" && pwd)"
experiment_dir="$(cd "$(dirname "$0")" && pwd)"
clangxx="$llvm_root/bin/clang++"
llc="$llvm_root/bin/llc"
ptxas="$cuda_root/bin/ptxas"
gcc_version="$(g++ -dumpversion)"
cxx_include="/usr/include/c++/$gcc_version"
arch_include="/usr/include/x86_64-linux-gnu/c++/$gcc_version"
temporary_dir="$(mktemp -d /tmp/cuda-compat-llvm-nvptx.XXXXXX)"
trap 'rm -rf "$temporary_dir"' EXIT

for required in "$clangxx" "$llc" "$ptxas" "$cxx_include/climits" "$arch_include"; do
  if [[ ! -e "$required" ]]; then
    printf 'required compiler input is unavailable: %s\n' "$required" >&2
    exit 2
  fi
done

llc_version="$("$llc" --version)"
printf '%s\n' "$llc_version" | sed -n '1,12p'
if ! printf '%s\n' "$llc_version" | grep -q 'nvptx64'; then
  printf 'LLVM build does not include the NVPTX target\n' >&2
  exit 2
fi
"$clangxx" -x cuda --cuda-path="$cuda_root" --cuda-gpu-arch=sm_52 \
  --cuda-device-only -S -O2 -std=c++17 \
  -isystem "$cxx_include" -isystem "$arch_include" \
  -o "$temporary_dir/vector_add.ptx" "$source_dir/vector_add.cu"
grep -E '^\.version|^\.target' "$temporary_dir/vector_add.ptx"
"$ptxas" -arch=sm_52 -o "$temporary_dir/vector_add.cubin" "$temporary_dir/vector_add.ptx"

g++ -std=c++17 -O2 -I"$cuda_root/include" \
  "$experiment_dir/vector_add_driver.cpp" -lcuda \
  -o "$temporary_dir/vector_add_driver"

for module in "$temporary_dir/vector_add.ptx" "$temporary_dir/vector_add.cubin"; do
  for device in 0 1 2; do
    CUDA_VISIBLE_DEVICES="$device" "$temporary_dir/vector_add_driver" "$module"
  done
done
