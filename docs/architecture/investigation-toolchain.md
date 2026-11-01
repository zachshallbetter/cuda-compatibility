# CUDA compatibility investigation toolchain

Status: initial research design (2026-11-01). No hardware validation implied.

## Principles
- Treat CUDA as separate host API/ABI, compiler/PTX/binary, runtime/driver, library, kernel execution and framework contracts.
- Never equate decompilation with disassembly, or a kernel launch with proof of correct GPU execution.
- Record documented facts, measured observations, hypotheses and unsupported claims separately.
- Use version-pinned, repeatable commands; record tool versions, hashes, GPU UUID/PCI ID, driver, toolkit, OS and architecture.
- Preserve the Titan Node three-GPU inference service. Run intrusive tests on an isolated GPU only after qualification (#23).
- A modern tool being installable does not mean it supports GM200. Maintain an explicit per-tool support matrix.

## Static extraction (#37)
- Source and headers: versioned header inventories, symbol/function/structure signatures, preprocessor guards, deprecation markers; use reproducible API/ABI diffing.
- Host binaries: `readelf`, `nm`, `objdump`, `ldd` (caution: do not run untrusted binaries), `libabigail/abidiff`, `c++filt`; inspect sonames, dependencies, exports and type layouts. Prefer `readelf -d` over executing unknown binaries.
- GPU binaries: `cuobjdump` for embedded fatbin/PTX/cubin and SM inventory; `nvdisasm` for SASS disassembly; `nvprune` only in controlled copies; identify version and architecture limitations.
- Disassembly does not recover original CUDA source or guaranteed high-level intent. PTX is not guaranteed present in shipping binaries.
- Record source licensing and redistribution constraints; do not upload proprietary binary payloads to issue logs.
- Produce content-hashed artifact manifests and structured extraction errors.

## Dynamic tracing and debugging (#38)
- Capture runtime/driver calls, library loads, allocation/copy/launch, streams/events, return codes and failure stage.
- CUPTI Activity/Callback tracing is candidate instrumentation; version and GM200 support must be experimentally confirmed. CUPTI Event/Metric interfaces change across versions.
- Qualify `cuda-gdb`, Compute Sanitizer and version-appropriate legacy `nvprof`/`nvvp` separately; newer Nsight Compute/Systems versions may not support Maxwell. Never promise unsupported counters.
- Establish CPU-only, legacy-CUDA native GPU and modern-workload failure baselines. Correlate launch to actual execution and numerical output.

## Numerical correctness (#39)
- Golden CPU/reference and known-working legacy CUDA baselines; seeded differential and property-based tests.
- Include FP32/FP16 conversions, rounding, ULP/relative/absolute tolerances, NaN/Inf, denormals, overflow, deterministic expectations, aliasing, strides, odd shapes, negative tests, memory safety and stream ordering.
- Verify explicit GPU execution and expose every CPU fallback; reduce failing inputs to minimal reproducers.

## Performance and topology (#40)
- Separate GPU event elapsed time from end-to-end host latency, compilation, copies, allocation and synchronization.
- Repeat with warmups, percentiles and variance; record clock/temperature/power and co-tenant interference.
- Measure H2D/D2H, D2D, PCIe, P2P, launch overhead, VRAM, kernel throughput, 1/2/3/4-card scaling; validate peer access instead of assuming SLI bridges enable CUDA P2P.
- Use NVML/`nvidia-smi` for device identity/telemetry, with version-specific availability. Never infer instruction counts from utilization.

## Evidence and reporting (#41)
Each claim should link: source URL; exact component and version; binary/header hash; tool invocation and version; device/driver environment; failure category; strategy; correctness results; performance results; run ID; confidence; related issue. Publish JSON and human-readable Markdown/HTML, with redaction of secrets and local paths.

## Failure taxonomy
Discovery → load/link → initialization → capability gate → compilation/JIT → dispatch → launch → synchronization → numerical result → performance regression. Report the earliest observed failure and retain subsequent errors.

## Execution order
1. #32 CUDA 12.9→13.0 source-backed ecosystem diff, supported by #37 extraction.
2. #38 trace/debug support qualification on physical GM200.
3. #39 differential correctness and #40 performance baselines.
4. #41 unified evidence ledger; integrate into #24–#35.
5. Only then expand PTX (#30) and SASS (#31) research according to observed gaps.

## References
- CUDA Toolkit 12.9 notes: https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html
- CUDA Toolkit 13.0 notes: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
- CUDA binary utilities: https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- Compute Sanitizer: https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
- CUPTI: https://docs.nvidia.com/cupti/index.html
- Nsight Compute release notes: https://docs.nvidia.com/nsight-compute/ReleaseNotes/index.html

References are starting points, not a completed verification of tool support.
