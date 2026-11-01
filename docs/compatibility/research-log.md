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
