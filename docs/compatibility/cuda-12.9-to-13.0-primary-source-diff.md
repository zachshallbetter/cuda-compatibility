# CUDA 12.9 GA → CUDA 13.0 GA: primary-source change register (v0.1)

**Status:** Release-note verified only. No SDK binary extraction, symbol diff or GM200 execution tests performed. Updated 2026-11-01.

Primary sources:
- https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html

## Release components (exact versions from NVIDIA toolkit component tables)

| Component | 12.9 GA | 13.0 GA | Investigation |
|---|---|---|---|
| cudart | 12.9.37 | 13.0.48 | runtime API, ABI, symbol map |
| NVCC | 12.9.41 | 13.0.48 | sm_52 target support, compilation flags |
| NVRTC | 12.9.41 | 13.0.48 | JIT API, supported architectures |
| CUPTI | 12.9.19 | 13.0.48 | tracing API and removed metric/event APIs |
| cuBLAS | 12.9.0.13 | 13.0.0.19 | algorithm enums, library kernels |
| cuFFT | 11.4.0.6 | 12.0.0.15 | errors, handles, GPU support |
| cuRAND | 10.3.10.19 | 10.4.0.35 | APIs and SM coverage |
| cuSOLVER | 11.7.4.40 | 12.0.3.29 | math mode, solvers, dependencies |
| cuSPARSE | 12.5.9.5 | 12.6.2.49 | 64-bit SpGEMM, pre-Turing removal |
| NPP | 12.4.0.27 | 13.0.0.50 | legacy API removals |
| nvJPEG | 12.4.0.16 | 13.0.0.40 | removed encoder API |
| Thrust/CUB/libcu++ | 2.8.0 | 3.0.1 | header/template source compatibility |

**Note:** cuDNN and NCCL have separate release/distribution lifecycles; do not invent a matching toolkit-bundled version.

## Verified breaking and behavior changes

| ID | Surface | 13.0 change | Candidate compatibility response | Status |
|---|---|---|---|---|
| ARCH-001 | compiler + libraries | Maxwell/Pascal/Volta offline compilation and library support removed | legacy SM-targeted compilation, compatible library backend or own kernels | documented; binary test pending |
| DRV-001 | driver | CUDA 13.x minor-version compatibility minimum driver R580; CUDA 13.0 GA toolkit driver Linux >=580.65.06 | host driver/API compatibility study; no assumption CUDA 13 user-space runs on Maxwell | documented |
| RT-001 | runtime API | `cudaGraphKernelNodeCopyAttributes` prototype argument order changed to destination then source | compile-time/ABI-specific wrapper | documented |
| JIT-001 | nvJitLink | `nvJitLinkCreate` options parameter type changed | header/source adaptation; ABI validation | documented |
| BIN-001 | fatbin | Zstd replaces LZ4 as default compression | inspect/decompress artifact; no assumption old loader accepts it | documented |
| BIN-002 | fatbinary | `-image` removed; use `-image2`/`-image3` | build flag migration | documented |
| MEM-001 | driver/runtime | managed-memory discard APIs `cuMemDiscardBatchAsync` and `cuMemDiscardAndPrefetchBatchAsync` added | capability gate or semantics-preserving alternative only if proven | documented |
| MEM-002 | runtime | contextless loading behavior introduced | loader/initialization conformance tests | documented |
| API-001 | runtime | `cudaLaunchCooperativeKernelMultiDevice`, `cudaLaunchParam`, multi-grid group references removed | assess legacy API preservation separately; no universal substitute | documented |
| API-002 | runtime | deprecated `cudaDeviceProp` fields removed; most have attribute queries, cooperativeMultiDeviceLaunch has no replacement | structure-layout ABI audit, wrapper where possible | documented |
| API-003 | runtime | legacy texture/surface reference headers removed | source migration or compatibility headers if behavior verified | documented |
| TOOL-001 | tooling | nvprof and NVVP removed; CUPTI Event/Metric and some sampling APIs removed | qualify 12.9 legacy instrumentation on Maxwell, modern tracing separately | documented |
| LIB-001 | cuBLAS | `CUBLAS_GEMM_AUTOTUNE` supported in selected GEMM APIs | optional autotuning adapter with benchmark/caching semantics | documented |
| LIB-002 | cuFFT | `CUFFT_PLAN_NULL`, new dependency/JIT error codes; legacy errors removed; pre-Turing support removed | handle/error translation + older FFT implementation | documented |
| LIB-003 | cuSPARSE | 64-bit-index SpGEMM; pre-Turing support removed | 64-bit index backend or explicit unsupported; native legacy kernels | documented |
| LIB-004 | cuSOLVER | math-mode and emulation-strategy APIs added | truthful capability negotiation, fallback where valid | documented |
| LIB-005 | NPP | legacy non-`_Ctx` functions and `nppGetStreamContext` removed | context-aware adapter, preserving stream semantics | documented |
| LIB-006 | nvJPEG | `nvjpegEncoderParamsCopyHuffmanTables` removed | explicit API compatibility decision | documented |
| OS-001 | platform | Ubuntu 20.04 support dropped | isolated build environment; don't alter production OS | documented |

## Important known correctness caveats
- CUDA 13.0 release notes report a cuFFT correctness issue for half/bfloat16 size-1 strided R2C/C2R. A modern library is not automatically a reliable numerical oracle for every case.
- cuSPARSE notes nondeterministic behavior for `CUSPARSE_SPMM_CSR_ALG3` despite prior documentation. Preserve known issues in test contracts.

## Research questions not yet answered
- Which exact CUDA 13 library distributions contain executable `sm_52` images? Release policy says unsupported; inventory actual binaries rather than inferring precise contents.
- Can pinned CUDA 13 user-space components initialize with the actual Maxwell-supported host driver? Requires isolated tests.
- Which CUDA 12.9 binary modules load under each target driver, and which ABI boundaries permit safe adapters?
- Can a modern PyTorch workload be intercepted below its dispatcher without changing application code?
- Which CUDA 13 PTX constructs can be lowered with a legacy-compatible backend?

## Next experiments
1. Acquire official 12.9 GA and 13.0 GA headers and toolkit artifacts in an isolated environment with license-aware storage.
2. Generate machine-readable header/symbol/ELF/SM-target diffs and provenance manifests (#37).
3. Qualify actual GM200 profiler/debugger compatibility (#38).
4. Run smallest correct CUDA 12.9 `sm_52` kernel, then modern workload failure capture (#26).
5. Feed each observed gap to capability registry and adapter/kernel backlog (#24–#35).

No claim of complete CUDA 13 compatibility is warranted by this register.
