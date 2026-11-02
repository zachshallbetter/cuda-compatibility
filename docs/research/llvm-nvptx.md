# LLVM NVPTX Maxwell experiment

**Evidence status:** Reproduced on the reference host, 2026-10-31.<br>
**Scope:** LLVM 23.1.4 CUDA frontend → PTX 8.4 (`sm_52`) → CUDA 12.4.131 `ptxas` and R580 driver; vector addition only.

## Environment

- Linux `reference GPU host`, x86_64, Ubuntu 22.04 userspace, glibc 2.35.
- Three GTX TITAN X GM200 devices, compute capability 5.2.
- NVIDIA driver 580.178.04 and CUDA Toolkit 12.4.131.
- Official LLVM 23.1.4 Linux x86_64 release archive, unpacked under `/tmp`; it registers both `nvptx` and `nvptx64` targets.
- The LLVM `.tar.zst` artifact passed offline GitHub attestation verification against `llvm/llvm-project`; local and remote SHA-256 matched: `efdd1dbd089b99bfcfcde7b8eb1236163f2e88df5a981cca4f783ec9cb5e0378`.

## Reproduction

The experiment uses the native baseline kernel in [`vector_add.cu`](../../experiments/maxwell/vector_add.cu) and the driver API runner in [`run_vector_add.sh`](../../experiments/llvm-nvptx/run_vector_add.sh). On the reference host:

```sh
LLVM_ROOT=/path/to/LLVM-23.1.4-Linux-X64 \
CUDA_ROOT=/usr/local/cuda-12.4 \
bash experiments/llvm-nvptx/run_vector_add.sh
```

The runner compiles device code with Clang for `sm_52`, prints the generated PTX version and target, assembles it with the existing CUDA 12.4 `ptxas`, then loads both the PTX and cubin through the CUDA Driver API. It runs each artifact in a separate process on each physical GPU and removes its temporary files on exit.

On this Ubuntu host, Clang's CUDA device compilation did not discover the host `libstdc++` headers automatically and initially failed to find `<climits>`. The runner supplies the installed GCC major-version C++ include directories explicitly. This is an environment-specific toolchain setup detail, not a source change or a CUDA limitation.

## Observed result

Clang emitted `.version 8.4` and `.target sm_52`. CUDA 12.4 `ptxas` accepted the PTX. The R580 driver loaded both the PTX module and the assembled cubin. Each run compared all 1,048,576 integer outputs; all six runs passed: three PTX-JIT runs and three cubin runs, one on each physical GPU.

`CUDA_VISIBLE_DEVICES` mapped each selected physical card to logical device 0 inside its own process.

## Limits

This proves one simple CUDA kernel can be compiled by this LLVM build for Maxwell, assembled by CUDA 12.4, and executed correctly through the R580 driver. It does not prove arbitrary CUDA C++ coverage, compatibility with all PTX 8.4 features, CUDA 13 runtime/API compatibility, PyTorch operator coverage, library compatibility, concurrent multi-GPU execution, peer access, or performance parity. The NVIDIA driver and CUDA 12.4 assembler remain user-installed dependencies; no NVIDIA files were redistributed or modified.
