# Running PyTorch's CUDA 13 wheel on a 2015 Maxwell GPU, and where it stops

**State:** Published at https://cudacompat.com/articles/pytorch-cuda-13-on-maxwell.html on 2026-11-03. **Source revision:** `main` at the commit that adds this file. **Canonical link:** https://cudacompat.com/
**Next action for readers:** run the [public example](https://github.com/zachshallbetter/cuda-compatibility/tree/main/examples/sm52-mlp) and send a [reproduction report](https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml).

Every factual sentence below names its evidence record in `.agents/records/evidence.jsonl`. Keep those IDs when editing, and remove a sentence rather than soften its limit.

---

CUDA 13 dropped support for Maxwell, Pascal and Volta GPUs. The official PyTorch 2.14 CUDA 13 wheel is built for compute capability 7.5 and newer. On a GTX TITAN X (GM200, compute capability 5.2) it still imports and sees the card, but real work fails: matrix multiplication stops at `cublasCreate` with `CUBLAS_STATUS_ARCH_MISMATCH`, and ReLU reports that no kernel image is available (`CLC-006`, `CLC-009`).

CUDA Compatibility is an experiment in getting a narrow, checked subset of that wheel working on the older card without changing the driver, the CUDA install or the wheel itself. It is independent research, not affiliated with NVIDIA.

## How it works

`cuda-legacy run -- app.py` starts one Python process with three pieces in place, built fresh in a temporary directory and removed when the process exits (`CLC-010`):

1. **Kernels compiled for the real card.** LLVM/Clang 23.1.4 compiles the project's kernels to PTX for `sm_52` at launch (`CLC-004`, `CLC-007`). A small CUDA Driver API bridge loads that PTX and launches it on PyTorch's current stream (`CLC-008`).
2. **PyTorch registrations.** Through `torch.library`, the adapter registers native Maxwell implementations for six operation families: addition, arange, embedding, layer normalization, matrix multiplication and ReLU. Each has explicit dtype, layout and size limits (`CLC-025`).
3. **Bounded CPU fallback.** Eleven more operations, and any operation that fails with a recognized "wrong architecture" error, are copied to the CPU, computed there and copied back. Each copy is capped at 64 MiB in and 64 MiB out. Operations that mutate or alias their inputs refuse to fall back rather than risk wrong results (`CLC-025`).

A fourth piece is a version facade. One CUDA Runtime query, `cudaRuntimeGetVersion`, returns 13000 inside that process only (`CLC-005`). The device name and compute capability are left untouched. The project never pretends the card is newer than it is.

## What has been shown

- On all three reference cards, an operator suite and a small two-layer MLP matched CPU results (`CLC-009`).
- On one card, an owner-supplied transformer checkpoint matched CPU output across six batch and sequence configurations (`CLC-011`). The checkpoint is not public.
- On two cards, first-order gradients and seven SGD steps of a 2×2 linear→ReLU→sum model matched the CPU reference (`CLC-025`).
- A public example, run from release `v0.1.0` in a new environment, reproduced each operation's route and CPU comparison on two cards with identical output (`GH-50-public-example-sm52-mlp`).

## Where it stops

- Creating a tensor directly on the GPU (`torch.ones(..., device="cuda")`) still fails: no Maxwell kernel exists, and there is no GPU input to fall back from (`GH-50-public-example-sm52-mlp`). *Update, 0.2.2:* the adapter now builds such tensors on the CPU and copies them over (#113).
- Native matrix multiplication stops at 256 in any dimension and refuses larger inputs rather than falling back (`GH-50-public-example-sm52-mlp`).
- Convolution works in one tested classifier only by running on the CPU (`CLC-012`).
- General cuDNN, cuBLAS, NCCL or Triton support, broad autograd, other optimizers, and performance parity are not established (`CLC-CLAIM-008`).
- The launcher requires Linux x86_64 and only `sm_52` GPUs visible. Other Maxwell variants and Pascal cards are unverified (`CLC-CLAIM-006`).
- The tiny training run took 2.360 ms and 7.296 ms per step on two cards. Those numbers are diagnostic, not a benchmark (`CLC-025`).

## Why publish something this narrow

Each limit above is a specific missing kernel or library path, not a vague "may not work". That makes it a work list. The engineering tickets track the blocker corpus and the replacement kernels, and nothing gets claimed until it has been reproduced on hardware.

## Try it

You need Linux, a GM200 or other compute capability 5.2 card, your own NVIDIA driver, CUDA 12.4 headers and LLVM 23.1.4. The project ships no NVIDIA files. The [public example](https://github.com/zachshallbetter/cuda-compatibility/tree/main/examples/sm52-mlp) gives the exact commands and expected output. Pass or fail, a [reproduction report](https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml) is the most useful thing you can send.
