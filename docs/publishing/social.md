# Drafts: short videos, LinkedIn and X

**State:** Drafts, none published. Each draft carries one next action and the canonical link. All facts come from [the article draft](article.md) and keep its evidence IDs. Before posting, re-check every number against `.agents/records/evidence.jsonl` at the revision you post from, and record the placement in the ledger (see [README](README.md)).

## Reveal short (about 45 seconds)

Screen capture of a real terminal on the reference machine; no staged output.

1. **0–5 s.** On screen: `nvidia-smi` showing GTX TITAN X, compute capability 5.2. Voice: "A 2015 GPU. PyTorch's CUDA 13 wheel says it's too old."
2. **5–15 s.** Run `python route_check.py` with the stock wheel. On screen: the `CUBLAS_STATUS_ARCH_MISMATCH` line. Voice: "Matrix multiply fails before it starts."
3. **15–30 s.** Run the same script through `cuda-legacy run`. On screen: the route list (`llvm-sm52-mm-f32`, `llvm-sm52-relu-f32`, `cpu-fallback-softmax`) and `matches_cpu: true`. Voice: "Same wheel, same card. Kernels compiled for the real chip; each result checked against the CPU."
4. **30–40 s.** On screen: the two rejected cases. Voice: "It also tells you where it stops."
5. **40–45 s.** Card: "Run it yourself: cudacompat.com". **Next action:** run the public example.

Evidence: `CLC-009-pytorch-fp32-mlp-sm52`, `GH-50-public-example-sm52-mlp`.

## Limitation short (about 30 seconds)

1. **0–10 s.** Run `torch.ones(2, 2, device="cuda")` under the adapter. On screen: "no kernel image is available". Voice: "This still fails. Here's why."
2. **10–22 s.** Diagram: the CPU fallback only engages when an operation has GPU inputs to copy back; a constructor has none. Voice: "Fallback needs something to copy. Creating memory on the card needs a real Maxwell kernel, and there isn't one yet."
3. **22–30 s.** Card: "Every gap is a ticket. Send a reproduction report: cudacompat.com". **Next action:** file a reproduction report.

Evidence: `GH-50-gpu-constructor-no-fallback` in `.agents/records/negative-results.jsonl`.

## LinkedIn

> PyTorch's official CUDA 13 wheel doesn't run on Maxwell-era GPUs. I've been working on an experiment that gets a narrow, checked part of it running on a 2015 GTX TITAN X without touching the driver, the CUDA install or the wheel.
>
> How: LLVM compiles kernels for the real chip at launch, PyTorch routes six operation families to them, and eleven more run on the CPU with a hard 64 MiB cap. Every result is compared with the CPU.
>
> What it is not: general CUDA 13 support. Creating tensors directly on the GPU still fails, convolution runs on the CPU, and there's no performance claim. The site lists each limit with the record behind it.
>
> If you have a compute capability 5.2 card, the public example takes a few commands, and a pass/fail report helps more than a like: https://cudacompat.com/

**Next action:** run the example and send a report. Evidence: `CLC-025-sm52-operator-training`, `GH-50-public-example-sm52-mlp`, `CLC-CLAIM-008`.

## X thread

1. PyTorch's CUDA 13 wheel won't run on Maxwell GPUs. I got a narrow, checked slice of it working on a 2015 GTX TITAN X, without changing the driver, the CUDA install or the wheel. Limits first, then how. 🧵 https://cudacompat.com/
2. Stock wheel on the card: matrix multiply fails at cublasCreate with CUBLAS_STATUS_ARCH_MISMATCH. ReLU has no kernel image.
3. Through `cuda-legacy run`: LLVM 23.1.4 compiles kernels for sm_52 at launch. Six op families run natively, eleven fall back to the CPU (64 MiB cap), and each result is compared with the CPU.
4. Only one CUDA call is spoofed: the runtime version query, in that one process. Device name and compute capability stay real.
5. Still broken: GPU tensor constructors, native matmul above 256 in any dimension, GPU convolution, general cuDNN/cuBLAS/NCCL/Triton. No performance claims.
6. Have a compute capability 5.2 card? Run the example and file a pass/fail report: https://github.com/zachshallbetter/cuda-compatibility/tree/main/examples/sm52-mlp

**Next action:** run the example. Evidence: `CLC-005-runtime-version-facade`, `CLC-009-pytorch-fp32-mlp-sm52`, `CLC-025-sm52-operator-training`, `GH-50-public-example-sm52-mlp`.

## Hacker News

Not drafted. HN's guidelines ask people not to post generated text, so the owner writes any Show HN title and first comment. Use the article's facts. Before posting, check the eligibility row in [channels.md](channels.md).
