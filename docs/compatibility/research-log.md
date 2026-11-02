# CUDA 12.9 → 13.0 research log

Status: preliminary documentation-level findings. Binary-level and runtime-level verification pending.

## Baseline
Compare **CUDA Toolkit 12.9 GA** against **CUDA Toolkit 13.0 GA**. Record exact independent library versions; do not conflate toolkit releases with component releases or later CUDA 13.x point updates.

| Research item | Current status | Required evidence |
|---|---|---|
| Maxwell/Pascal/Volta offline compilation and library support removed in CUDA 13 | Documented in NVIDIA release notes | Verify actual NVCC target behavior and shipped library SM images |
| Maxwell `sm_52` still compilable with legacy 12.x toolkit | Documented support | Build, inspect and execute test kernel on GM200 |
| Driver/toolkit version boundary | Documented release policy | Check host driver version and supported API behavior |
| Runtime/driver API additions and removals | Partial | Header/ABI/symbol diff |
| Library API/ABI and kernel image coverage | Unverified | Per-component binary inventory |
| PTX/fatbin compatibility | Unverified | Extract real samples and identify PTX ISA, SM and compression |
| Debugger/profiler/sanitizer Maxwell support | Unverified | Version-pinned support matrix and hardware probes |
| Modern PyTorch unchanged workload | Unverified | Isolated reproduction with failure-stage tracing |
| Cross-domain primitive coverage | Planned | GPU differential tests and benchmarks |

## Important distinctions
1. CUDA 13 not generating `sm_52` is a compiler support removal, not a change in the arithmetic Maxwell can physically execute.
2. A modern CUDA API wrapper does not make an incompatible cuBLAS/cuDNN binary contain Maxwell kernels.
3. Older numerical libraries behind adapters require explicit ABI, dynamic linking, streams, error, precision and workspace verification.
4. CUDA 12.9 compiling a Maxwell kernel does not mean CUDA 13 toolchains compile Maxwell kernels.
5. Installed aggregate GPU VRAM is not a single allocation; SLI graphics linkage does not establish CUDA P2P.
6. 'Compatibility' must be scoped to verified operations, library versions, workload shapes and actual devices.

## Required output schema (draft)
`component, component_version, toolkit_release, symbol_or_operation, change_type, sm_requirement, binary_artifact_hash, source_reference, evidence_status, compatibility_strategy, correctness_test, performance_baseline, issue, known_limitations`.

## Initial priorities
- API and ABI extraction (#37, #32).
- Library/SM audit (#33).
- Debugging and profiling support matrix (#38).
- Reference correctness and benchmark harnesses (#39, #40).
- Evidence ledger (#41) and architectural decision records (#36).

See [investigation toolchain](../architecture/investigation-toolchain.md) and issue #32 research note.

## 2026-11-01 reconciliation with existing implementation evidence

**Important correction:** the preceding initial status table was a planning baseline, not a current-state inventory. The repository already contains substantial physical GM200 and official PyTorch 2.14.0+cu130 evidence. See the [README](../../README.md), [Maxwell baseline](../research/maxwell-baseline.md), [PyTorch CUDA 13 research record](../research/pytorch-cu130-maxwell.md), and [architecture](../architecture.md) as authoritative implementation evidence.

- Three GM200 cards and driver 580.178.04: native CUDA **12.4**, not 12.9, vector-add correctness was verified per GPU. CUDA 12.9 artifact acquisition remains pending.
- Official PyTorch 2.14.0+cu130 imports, enumerates genuine CC 5.2 cards, allocates and copies tensors. Stock tensor add fails with `no kernel image`; stock matrix multiplication fails at `cublasCreate` with `CUBLAS_STATUS_ARCH_MISMATCH`.
- Existing `cuda-legacy run` launches a child process with an opt-in dispatcher adapter, LLVM 23.1.4-generated `sm_52` PTX, a CUDA Driver API bridge, and a scoped runtime-version facade. It has selected GPU-native operations and capped CPU fallbacks. It is not a general-purpose ABI interposer or universal CUDA 13 execution path.
- The project reports a bounded transformer-checkpoint comparison, a small MLP, selected float16/strided kernels and tiny first-order autograd + SGD test, with stated precision and workload limitations. Read the experiment log before making broader claims.
- `cuda-legacy inspect`, `doctor`, `audit`, `profiles` and `run` already exist. Product ticket #42 must enhance and test these, not replace them with a parallel `cuda-compat` implementation.
- Physical P2P is **not established**: observed topology reports PHB, and older prose describing SLI/P2P should not be promoted to a verified capability.

**Next action:** merge the primary-source CUDA diff and tooling proposals with the existing research ledger; avoid duplicate probes and clearly distinguish tests already run from new work.
