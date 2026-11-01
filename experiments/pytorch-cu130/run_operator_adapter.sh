#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/../.." && pwd)"
experiment_dir="$(cd "$(dirname "$0")" && pwd)"
llvm_root="${LLVM_ROOT:-/tmp/cuda-compat-llvm-23.1.4/LLVM-23.1.4-Linux-X64}"
cuda_root="${CUDA_ROOT:-/usr/local/cuda-12.4}"
python_bin="${PYTHON_BIN:-/tmp/cuda-compat-pytorch-cu130-venv/bin/python}"

for required in "$python_bin" "$llvm_root/bin/clang++" "$cuda_root/include/cuda.h"; do
  if [[ ! -e "$required" ]]; then
    printf 'required tool or header is unavailable: %s\n' "$required" >&2
    exit 2
  fi
done

export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
for operation in mm relu; do
  printf 'mode=stock_cuda13_builtin operator=%s\n' "$operation"
  env -u CUDA_COMPAT_RUNTIME_VERSION -u CUDA_COMPAT_BRIDGE \
    -u CUDA_COMPAT_VECTOR_ADD_PTX -u CUDA_COMPAT_ENABLE -u LD_PRELOAD \
    CUDA_COMPAT_BASELINE_OPERATOR="$operation" \
    "$python_bin" "$experiment_dir/builtin_operator_probe.py"
done
printf 'mode=public_launcher_virtual_cuda13_profile\n'
"$python_bin" -m cuda_legacy_compat.cli run --llvm-root "$llvm_root" \
  --cuda-root "$cuda_root" "$experiment_dir/operator_probe.py"
