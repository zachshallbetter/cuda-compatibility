# Public example: a small MLP on an `sm_52` GPU, with its execution routes

This example takes one supported workload from the [claim register](../../docs/DPP_IMPLEMENTATION.md) and makes every part recoverable: the software, the machine, the commands, the expected output, how each result is compared with the CPU, which route each operation took, and two documented failures. It supports claims `CLC-CLAIM-002`, `CLC-CLAIM-004`, `CLC-CLAIM-005` and `CLC-CLAIM-008`. It is project evidence run by the author, not an independent reproduction.

## What it runs

[`route_check.py`](route_check.py) builds a two-layer float32 network from seeded random inputs (`8×16 → 32 → ReLU → 8`, then softmax). It runs each operation on the GPU, records the route the adapter reports through `torch_adapter.last_path()`, and compares that result with the CPU result of the same operation on the same inputs.

| Operation | Expected route | Comparison with CPU |
|---|---|---|
| `mm` 8×16 @ 16×32 | `llvm-sm52-mm-f32` (native Maxwell kernel) | Within the float32 dot-product bound `2·n·u·(|A|@|B|)`, `u = 2⁻²⁴` |
| `add` bias 8×32 | `llvm-sm52-binary-f32` (native) | Identical |
| `relu` 8×32 | `llvm-sm52-relu-f32` (native) | Identical |
| `mm` 8×32 @ 32×8 | `llvm-sm52-mm-f32` (native) | Within the same bound |
| `softmax` 8×8 | `llvm-sm52-softmax-f32` (native) | Within the float32 softmax bound `(n+2)·u·|softmax|` |
| `mm` 1×257 @ 257×1 | `llvm-sm52-mm-f32` (native tiled kernel; past the old 256-dimension limit) | Within the same bound |
| `torch.ones(..., device="cuda")` | `llvm-sm52-fill-f32` (native) | Identical |

The matmul bound is the standard forward error bound for an `n`-term float32 dot product, doubled because the CPU reference is also rounded. It was set before the passing run and not fitted to the data. An earlier run compared the whole chained network with fixed `rtol=1e-5, atol=1e-6`; one element of the second matmul differed by 7.63×10⁻⁶ and failed. That is a float32 summation-order difference, kept as a negative result in `.agents/records/negative-results.jsonl`.

No timing is measured or claimed.

## Tested configuration

| Fact | Value |
|---|---|
| Project | release `v0.1.0` (commit `de592549`), installed from source into a new virtual environment; `route_check.py` from this directory |
| GPU | NVIDIA GeForce GTX TITAN X (GM200), compute capability 5.2; physical GPUs 1 and 2 |
| Driver | 580.178.04 |
| OS / Python | Linux x86_64 / Python 3.10.12 |
| Framework | official `torch 2.14.0+cu130` wheel; `nvidia-cuda-runtime 13.0.96` |
| Compiler | LLVM/Clang 23.1.4 Linux X64 (archive sha256 `efdd1dbd…0378`), `g++` 11 |
| Headers | CUDA 12.4 development headers at `/usr/local/cuda-12.4` |

## Reproduce

Use Linux x86_64 with only `sm_52` GPUs visible. Install your own NVIDIA driver, CUDA 12.4 headers and LLVM 23.1.4; this project ships none of them.

```sh
git clone --branch v0.2.11 https://github.com/zachshallbetter/cuda-compatibility
python3.10 -m venv venv
venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
venv/bin/python -m pip install ./cuda-compatibility
curl -LO https://raw.githubusercontent.com/zachshallbetter/cuda-compatibility/main/examples/sm52-mlp/route_check.py
export CUDA_VISIBLE_DEVICES=0      # one sm_52 card with at least ~700 MiB free
venv/bin/cuda-legacy setup          # finds or downloads LLVM 23.1.4, saves it, runs doctor
venv/bin/python route_check.py     # stock wheel
venv/bin/cuda-legacy run route_check.py
```

Expected results:

- `doctor` prints `ready` and exits 0.
- The stock run prints `"fails": true` with `CUBLAS_STATUS_ARCH_MISMATCH when calling cublasCreate(handle)`, and exits 0 because that failure is the expected outcome. The stock wheel also warns that it has no kernels for compute capability 5.2.
- The adapter run prints the JSON in [`expected-output.json`](expected-output.json) and exits 0. Error magnitudes can differ on other hardware; the routes, the pass/fail fields and the two rejections should not.

Either run exits 1 when its expected outcome is not observed, and 2 when CUDA is unavailable.

## Recorded reproduction

On 2026-11-03 the steps above ran on the project's reference host in a new virtual environment under a task-scoped `/tmp` directory, which was removed afterwards. `doctor` was ready; the stock run reproduced the cuBLAS failure; the adapter run passed on physical GPUs 1 and 2 with byte-identical output (sha256 `b65ccafb…686b`). GPU 0 was not run because it had 313 MiB free. The record is `GH-50-public-example-sm52-mlp` in [`.agents/records/evidence.jsonl`](../../.agents/records/evidence.jsonl).

Still open: this is the author's machine. No independent tester on separate hardware has reported a run yet.

## Report your result

Open a [reproduction report](https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml). Failed and partial runs are as useful as passing ones.
