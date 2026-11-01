# PyTorch 2.14 CUDA 13 wheel on Maxwell

**Initial package/stock-wheel evidence:** Reproduced on the reference host, 2026-10-31.<br>
**Latest workload evidence:** Updated 2026-11-01; CLC-011 remains in progress and CLC-012 is verified.<br>
**Scope:** Official PyTorch 2.14.0 CUDA 13.0 wheel; Python 3.10.12; three GM200 GPUs; ordinary wheel behavior with and without the CLC-005 runtime-version facade.

## Installation

Installed in a disposable virtual environment under `/tmp` on `reference GPU host` using PyTorch's published command pattern:

```sh
python3.10 -m venv /tmp/cuda-compat-pytorch-cu130-venv
/tmp/cuda-compat-pytorch-cu130-venv/bin/python -m pip install \
  torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

The installed wheel reports `torch=2.14.0+cu130` and `torch.version.cuda=13.0`. Its CUDA dependencies include CUDA Runtime 13.0.96, cuDNN 9.24.0.43, and NCCL 2.30.7. See the [official PyTorch 2.14 install commands](https://pytorch.org/get-started/previous-versions/) and [published architecture matrix](https://github.com/pytorch/pytorch/blob/v2.14.0/RELEASE.md).

## Results

With each physical GPU selected separately by `CUDA_VISIBLE_DEVICES`, PyTorch imported, reported CUDA available, enumerated the actual GTX TITAN X at compute capability 5.2, allocated CUDA tensors, and copied host data to the device. The built-in tensor addition then failed with:

```text
torch=2.14.0+cu130 built_cuda=13.0
cuda_available=True device_count=1
device=0 name=NVIDIA GeForce GTX TITAN X compute=5.2
PASS device=0 host_to_device elements=1024
FAIL stage=CUDA tensor addition: AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

The same result occurred on physical devices 0, 1, and 2. Running with `CUDA_COMPAT_RUNTIME_VERSION=13000` and the CLC-005 `LD_PRELOAD` facade produced the same failure. The runtime-version response is therefore not the blocking layer; the wheel has no executable kernel image for `sm_52` for this operation. PyTorch's own warning lists supported architectures beginning at `sm_75`, consistent with its [published matrix](https://github.com/pytorch/pytorch/blob/v2.14.0/RELEASE.md).

## First compatibility adapter

`experiments/pytorch-cu130/run_operator_adapter.sh` generates PTX from the existing CUDA vector-add kernel with LLVM 23.1.4, builds a small CUDA Driver API bridge, and installs an explicit process-local Python dispatcher override for `aten::add.Tensor`. The adapter requires the visible devices to be `sm_52` and does not modify the PyTorch wheel.

Reproduce inside the disposable PyTorch environment after placing the verified LLVM release under `/tmp`:

```sh
PYTHON_BIN=/tmp/cuda-compat-pytorch-cu130-venv/bin/python \
LLVM_ROOT=/tmp/cuda-compat-llvm-23.1.4/LLVM-23.1.4-Linux-X64 \
bash experiments/pytorch-cu130/run_operator_adapter.sh
```

For each of the three cards, in both the unmodified runtime mode and with the CLC-005 runtime-version facade enabled:

- Contiguous `int32` `torch.add` used the LLVM-generated `sm_52` kernel; all outputs matched a CPU PyTorch reference.
- Integer overflow (`INT32_MAX + 1`) wrapped as expected.
- Contiguous `int64` `torch.add` used CPU PyTorch and copied the result back into a CUDA tensor; outputs matched the CPU reference.
- `float32` tensor addition failed explicitly as unsupported.

The PTX target was `.version 8.4` / `.target sm_52`. The stock PyTorch wheel still emits its warning that its packaged kernels start at `sm_75`; the adapter supplies only the tested `aten::add.Tensor` paths. It does not establish compatibility for scalar addition, `out=`, autograd, other operators, cuDNN, NCCL, Triton, or general PyTorch workloads, and no performance measurements were taken.

## Bridge hardening

CLC-008 removes the per-call module load/unload and `cuStreamSynchronize` from the int32 path. The bridge now retains one PTX module/function for each CUDA context for the process lifetime, rejects a different PTX path after that context is initialized, bounds the signed 32-bit kernel count, and launches on the stream PyTorch reports as current. Python dispatch diagnostics are thread-local. Empty int32 tensors return an empty result without launching a zero-sized grid.

The updated runner passed on physical GPUs 0, 1, and 2 in both the ordinary CUDA 13 runtime mode and the process-local runtime-version facade mode. It checked repeated module-cache use, rejection of a second valid PTX path in the same context, rejection of an oversized count, empty tensors, non-default-stream ordering, int32 overflow behavior, int64 CPU fallback equality, and explicit float32 rejection. Each probe process returned success and retained the true GTX TITAN X / CC 5.2 identity. LLVM emitted PTX 8.4 targeting `sm_52`; the downloaded LLVM archive matched the checksum of the attestation-verified CLC-004 artifact. Temporary source, wheel environment, LLVM extraction, and archive were removed using the host resource path.

This hardening does not broaden the supported operator set. The module cache is bounded to one PTX path per CUDA context and remains resident until process/context teardown; performance was not benchmarked. Clang emitted a non-fatal warning about selecting between installed GCC C++ include layouts.

## Float32 inference slice

CLC-009 adds `compat_kernels.cu` to the cached PTX module with float32 matrix multiplication and ReLU kernels, registered as opt-in CUDA implementations of `aten::mm` and `aten::relu`. Matrix multiplication accepts contiguous 2D float32 inputs with each dimension at most 256 and at most 16,777,216 multiply-accumulate steps. ReLU accepts contiguous float32 inputs up to 1,048,576 elements. Both run asynchronously on PyTorch's current stream. Empty output dimensions return empty tensors; a zero inner matrix dimension produces zeros. Other dtypes, non-contiguous inputs, oversized matrices, and autograd fail explicitly.

The stock CUDA 13 wheel was probed separately on each card. Built-in `mm` failed at `cublasCreate` with `CUBLAS_STATUS_ARCH_MISMATCH`; built-in `relu` failed with `no kernel image is available for execution on the device`. The adapter bypassed those built-in paths. CPU-reference checks covered matrix shapes `(3,5)@(5,4)`, `(1,7)@(7,1)`, `(2,0)@(0,3)`, and `(0,5)@(5,2)`; ReLU edge values including NaN, infinities, and signed zero; and a deterministic two-layer MLP with shapes `8x16 → 8x32 → ReLU → 8x8`. The MLP ran on a non-default stream and matched CPU PyTorch at `rtol=1e-5, atol=1e-6`.

The combined runner passed all assertions on physical GPUs 0, 1, and 2 in both ordinary CUDA 13 runtime and version-facade modes. LLVM emitted `.version 8.4` / `.target sm_52`; physical identity remained GTX TITAN X / CC 5.2. The earlier int32 addition checks, module caching, alternate-path rejection, count bounds, empty addition, stream ordering, and int64 CPU fallback also continued to pass. Temporary PyTorch, LLVM, source-transfer, and build files were removed through the host resource broker.

This is a synthetic small MLP slice, not a packaged pretrained model or general inference stack. Matrix multiplication uses a simple scalar-output kernel and no performance benchmark was taken. CUDA tensor constructors that invoke unsupported built-in fill kernels are not intercepted; test inputs use CPU tensors copied to the device. cuDNN, NCCL, Triton, autograd, bias operations, convolution, attention, and larger matrices remain outside this evidence.

## Owner-supplied checkpoint integration probe

One owner-supplied transformer checkpoint was exercised with the official PyTorch 2.14.0 CUDA 13.0 wheel through the packaged launcher on broker-selected GPU 2 (`sm_52`). Eleven returned tensors matched the CPU reference at `rtol=1e-4, atol=1e-5` across six bounded batch/sequence configurations. The largest case measured 114.881 ms median latency, 8,914 tokens/s, and 142,024,704 bytes peak allocated GPU memory. Selected operations used native Maxwell kernels; other covered operations used the documented 64 MiB-capped, inference-only CPU fallback.

This was a one-off integration experiment, not a product feature or general PyTorch support claim. The model source, checkpoint, tokenizer, configuration, artifact digests, and local-only probe are excluded from the public source snapshot. That checkpoint was tested only on GPU 2; per-card checkpoint validation and broader workload coverage remain unverified. A separate, much smaller first-order training slice is documented under CLC-025 below.

## CLC-025: stream, dtype, and training expansion

The updated package was rebuilt and exercised through the packaged launcher on broker-selected GPU 2 and then on physical GPU 1, both GTX TITAN X GM200 (`sm_52`), with the official `torch 2.14.0+cu130` wheel and LLVM 23.1.4. Stock-wheel matrix multiplication still fails at `cublasCreate` with `CUBLAS_STATUS_ARCH_MISMATCH`; stock ReLU still reports no kernel image. These are separate observed library/kernel gaps.

New native cases passed CPU comparisons on GPUs 1 and 2: contiguous float16 addition including NaN, infinities, signed zero, and a rounding tie (exact equality); transposed/strided float32 matrix multiplication; and scaled contiguous float32 in-place addition. Float16 addition is capped at 1,048,576 values. Matrix multiplication remains capped at 256 per dimension and 16,777,216 multiply-accumulate steps. In-place float32 addition is capped at 1,048,576 values. Existing cross-stream allocator-churn probes passed for float16 add, float32/int32 add, float32 matrix multiplication, ReLU, and layer normalization. Layer normalization's larger positive-offset case had maximum absolute error `3.34e-5` and uses `atol=5e-5`; its ordinary case remains `atol=1e-6`.

First-order gradients and an optimizer step passed for a 2×2 float32 `linear → ReLU → sum` model. Seven SGD updates with `foreach=False` matched CPU reference gradients and final parameters. The registered CPU fallback preserves the graph through framework host/device copies, and a custom CPU implementation supplies the missing `aten::linear_backward` behavior; the final scaled parameter update ran through the `sm_52` float32 `add_` kernel. The seven measured end-to-end steps had 2.360 ms median wall latency and 67,180,032 bytes peak allocated on GPU 2; a repeat on GPU 1 measured 7.296 ms median with the same peak allocation. Timing includes fallback transfers, CPU computation, and explicit synchronization; it is a tiny diagnostic workload, not a performance claim. Each fallback remains capped at 64 MiB of inputs and 64 MiB of outputs per operation.

### Runtime and library boundary inventory

| Observed path | Evidence | Current handling / boundary |
|---|---|---|
| `cudaRuntimeGetVersion` | Launcher facade probe returns `13000` in the child process | One interposed query; other runtime calls are forwarded, not broadly translated. |
| `cuCtxGetCurrent`, `cuModuleLoad`, `cuModuleGetFunction`, `cuLaunchKernel` | Native bridge implementation and successful `sm_52` kernel probes | Used to load cached LLVM PTX and launch on PyTorch's current stream. |
| `cuMemcpyDtoH` | Bounded fallback and embedding-index validation code | Explicit driver copy; some differentiable paths use PyTorch's device transfers instead. |
| cuBLAS initialization (`cublasCreate`) | Stock official wheel reports `CUBLAS_STATUS_ARCH_MISMATCH` on `mm` | Bypassed for the registered bounded float32 matrix kernel; other cuBLAS operations remain unverified. |
| cuDNN engine selection | Earlier convolution probe reports no engine for its CNN case | That convolution uses CPU fallback; general cuDNN compatibility is not implemented. |
| CUDA wheel and driver library files | `experiments/pytorch-cu130/library_path_probe.py` inspected `/proc/self/maps` after bounded matrix and convolution calls | Observed wheel paths under `site-packages/nvidia/cu13/lib/` for `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, and `libnvrtc.so.13`; under `site-packages/nvidia/cudnn/lib/` for cuDNN 9 libraries; under `site-packages/nvidia/nccl/lib/` for `libnccl.so.2`; plus `torch/lib/libtorch_cuda.so`. The installed driver mapped as `/usr/lib/x86_64-linux-gnu/libcuda.so.580.178.04`. |
| Missing CUDA kernel image | Stock ReLU failure and fallback-dispatch probes | Replaced only for registered kernels or eligible bounded CPU fallback cases. |
| NCCL, Triton, arbitrary runtime APIs | No supporting trace or workload evidence | Not mapped or claimed compatible. |

This is an implementation-boundary and loaded-library inventory, not a complete dynamic API call trace; a mapped library does not prove that every symbol or code path works. Paths above are from the task-scoped Titan Python 3.10 environment and may differ on user machines. CLC-025 float16, strided-matrix, cross-stream lifetime, and small training probes passed on GPUs 1 and 2; the process-map library inventory ran on GPU 2 only. GPU 0 was not tested in this increment because its free memory was below the 700 MiB test floor. Other optimizers, foreach updates, higher-order gradients, AMP, arbitrary autograd graphs, broad model training, and performance parity remain unverified.

## Convolutional workload probe

A second model family was exercised with the official CUDA 13 wheel on broker-selected GPU 2: a deterministic `Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → AdaptiveAvgPool2d → Flatten → Linear` classifier for input shape `2×3×32×32`. The stock cuDNN path failed with `GET was unable to find an engine to execute this computation`. After adding that exact cuDNN no-engine phrase to the architecture-failure classifier, the packaged launcher retried the operation on CPU; the final `(2, 5)` output matched the CPU reference at `rtol=1e-4, atol=1e-5`. This establishes one bounded inference case, not general cuDNN or convolution support; its convolution operations execute on CPU under the 64 MiB fallback limit.

Inference evidence is limited to GPU 2, two small model families, batch sizes up to 8, sequence lengths up to 128, and inference without autograd. The CNN test has one input shape and uses CPU fallback for convolution. Separately, the 2×2 first-order training slice in CLC-025 passed on GPUs 1 and 2; other training graphs, optimizers, broader architectures, larger fallbacks, other GPU cards, and independent API/library coverage remain unverified. CLC-011 remains in progress; CLC-012's bounded CNN probe is verified.


## CLC-010 packaged launcher

CLC-010 packaged the adapter and its CUDA/driver source in the Python wheel. On Linux x86_64, `cuda-legacy-compat run --llvm-root <LLVM-23.1.4> --cuda-root <CUDA-12.4> -- <script.py> [args...]` compiles PTX, the driver bridge, and the runtime-version facade into a temporary directory, then starts the caller's Python interpreter with an opt-in `sitecustomize` hook. The hook confirms all visible devices are `sm_52`, registers the compatibility implementations described above, and sets the runtime version facade to 13.0 while keeping hardware capability truthful. Child arguments, stdio, and exit status pass through; temporary build output is removed after exit. This packaging/launch path does not claim arbitrary PyTorch CUDA 13 model support.
