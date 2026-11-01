# Roadmap

## M0 — Reproducible project and native baseline

Project operating substrate, source-linked target records, deterministic inspector, and the CUDA 12.4 per-card correctness baseline. Status: complete for the initial slice; see `records/evidence.jsonl`.

## M1 — Compiler independence

Run upstream Clang/LLVM NVPTX on Linux, target the actual `sm_52` device, pass PTX to the available CUDA 12.x assembler/driver, and verify numerical output on each card. **Initial vector-add path complete:** LLVM 23.1.4 emitted PTX 8.4 for `sm_52`; CUDA 12.4 `ptxas` accepted it; the R580 driver passed both PTX-JIT and cubin runs on all three cards. See [LLVM NVPTX Maxwell experiment](research/llvm-nvptx.md). This is compiler-path evidence for one kernel, not general CUDA 13 or PyTorch compatibility.

## M2 — Newer source feature lowering

Select one concrete CUDA 13 source-language or PTX feature that does not require new execution hardware. Add a bounded lowering or API adapter with paired equivalence fixtures and GM200 execution evidence. Do not choose an operation before M1 identifies the useful boundary.

## M3 — Version/API compatibility facade

Prototype a process-local compatibility profile for one named application. **Version-query probe complete:** an opt-in `LD_PRELOAD` shim reports CUDA Runtime 13.0.0 while preserving the real R580 driver report and `sm_52` device properties; a CUDA 12.4 vector-add operation still passes on each reference GPU. See [runtime-version facade experiment](research/runtime-version-facade.md). This is only a spoofed version-query proof, not CUDA 13 API support. Next, inventory the target application's calls and add only backed runtime mappings/emulation; preserve actual device capability unless a specific behavior is implemented and validated.

## M4 — Framework/library adapter

Start with PyTorch 2.14 CUDA 13 and compare against the Maxwell-supported PyTorch 2.14 CUDA 12.6.3 profile. **Initial operator and toy-inference slice complete:** opt-in `aten::add.Tensor`, bounded contiguous float32 `aten::mm`, and float32 `aten::relu` replacements use LLVM-generated `sm_52` PTX. A deterministic two-layer MLP composition matched its CPU reference on all three GM200 cards. Stock-wheel `mm` fails in cuBLAS with `CUBLAS_STATUS_ARCH_MISMATCH`; stock-wheel `relu` reports no kernel image. CLC-010 packages the prototype behind an opt-in Linux `cuda-legacy-compat run` command. See [PyTorch CUDA 13 on Maxwell](research/pytorch-cu130-maxwell.md). **Owner-supplied checkpoint probe reached:** one transformer checkpoint passed CPU comparisons on broker-selected GPU 2 for six bounded batch/sequence cases, with all 11 outputs matching; the model and local-only probe are not included in the public source. A separate convolution/pooling classifier also matches its CPU result on GPU 2 after a narrowly recognized cuDNN no-engine fallback. The path combines native `sm_52` kernels with bounded inference-only CPU fallbacks; per-card validation remains open. TensorRT 11.4 and current vLLM remain outside the first target.

## M5 — Host installation and application integration

Turn the per-process prototype into a reversible, host-installed toolset that can launch qualified CUDA 13+ applications through explicit compatibility profiles. Start with Linux x86_64 and PyTorch; discover the host GPU, driver, toolkit, and application environment; report missing prerequisites; provide a scoped launch path, diagnostics, upgrade, and uninstall behavior. Keep proprietary driver/toolkit files user supplied and avoid system-wide loader injection. Acceptance requires a clean install and removal, clear audit/launch outcomes, and a documented route for applications that need native binaries rather than Python entry points. This milestone is not complete today: the package currently installs only into a selected Python environment and launches one Python script.

## M6 — Practical workload coverage

Expand from the current inference checkpoint to representative PyTorch inference and training workloads, dtypes, shapes, and model families. Implement Maxwell kernels where they improve device execution and bounded CPU fallbacks where semantics are preserved. Maintain a tested support matrix with correctness, unsupported cases, latency, throughput, memory use, and card-by-card evidence. Do not infer general compatibility from a single model.

## M7 — Multi-GPU scaling

Measure single-GPU and distributed/sharded variants across the three cards. Query peer access and topology at runtime; do not assume pooled memory. Compare end-to-end results, including transfer and synchronization overhead.

## M8 — Independent backend and binary research

Separately assess NVK/SPIR-V and limited fatbinary/PTX extraction. Keep these as experiments until they pass correctness and compatibility gates; arbitrary cubin translation remains out of scope absent a demonstrated viable method.
