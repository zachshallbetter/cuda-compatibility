# Maxwell native baseline

**Evidence status:** Reproduced on the reference host, 2026-10-31.<br>
**Scope:** CUDA 12.4 native `sm_52` build and sequential correctness run per GPU.

## Hardware and software observed

- Linux host `reference GPU host`, Ubuntu kernel `5.15.0-198-generic`, x86_64.
- Three NVIDIA GeForce GTX TITAN X / GM200 devices, compute capability 5.2, 12,288 MiB each.
- NVIDIA proprietary driver `580.178.04`.
- CUDA Toolkit 12.4.131 at `/usr/local/cuda-12.4`; `nvcc` absent from default SSH `PATH`, but direct path works and reports both `compute_52` and `sm_52`.
- `clang` and `llvm-config` absent from the target host's default `PATH`.
- `nvidia-smi topo -m` reports PHB between card pairs. This does not establish direct peer access or memory pooling.

The development machine is macOS on Apple Silicon and has no NVIDIA GPU or local `nvcc`. Docker daemon access on the target was denied, so no container toolchain was installed or run.

The Python 3.10.12 reference host successfully ran the project inspector from temporary copied source. It reported all three GM200 cards at compute capability 5.2, driver 580.178.04, 12,288 MiB each, and CUDA 12.4 `nvcc` with `sm_52` support. Auditing the PyTorch 2.14 CUDA 13 profile returned exit code 1 and identified each card as below the published `sm_75` minimum; its JSON separately reported that a local CUDA toolkit is not required merely to run the prebuilt package.

## Kernel and command

[`vector_add.cu`](../../experiments/maxwell/vector_add.cu) initializes two one-million-element integer vectors, launches a bounds-checked addition kernel, copies results back, and compares every element. [`run_native_baseline.sh`](../../experiments/maxwell/run_native_baseline.sh) compiles with CUDA 12.4 and `-arch=sm_52`, then runs a separate process selecting each device with `CUDA_VISIBLE_DEVICES`. It deletes the temporary executable at exit.

Reproduced on the target by copying the source and runner into `/tmp` and running:

```sh
bash /tmp/run_native_baseline.sh
```

Observed result:

```text
PASS device=0 elements=1048576 result=1048576
PASS device=0 elements=1048576 result=1048576
PASS device=0 elements=1048576 result=1048576
```

Each line used a different physical card through the one-device `CUDA_VISIBLE_DEVICES` mapping. This confirms separate per-card correctness. It does not claim simultaneous multi-GPU execution, peer-to-peer access, comparative performance, CUDA 13 compatibility, or LLVM-generated PTX compatibility.

## Negative / pending evidence

- CUDA Toolkit 12.9 was not installed; only configuration-common packages were present. CUDA 12.4 served as the available native baseline.
- CUDA Toolkit 13.x and Clang/LLVM NVPTX were not available on the host, so no local rejection or success from those compilers was captured.
- `nvidia-smi -q -d PCI` failed to parse the `PCI` display flag; this failed query supplies no PCI capability evidence.
- P2P must be measured with CUDA peer-query and transfer experiments; do not promote the repository's older “P2P bridged” prose without fresh evidence.

## Next experiment

Use an isolated Linux build environment with LLVM/Clang NVPTX, preserve the NVIDIA 12.4 assembler/driver as a separate downstream stage, compile equivalent vector-add source to PTX for `sm_52`, assemble it, and run the exact correctness checks on all three cards. Record the earliest incompatibility only if one is reproduced. Do not alter the host driver or system CUDA installation.
