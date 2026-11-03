# Project Intent

**Project ID:** `cuda-legacy-compat`<br>
**Project Name:** `CUDA Compatibility`<br>
**Status:** Active

## Originating intent

CUDA Compatibility extends the useful life of older NVIDIA GPUs through independently maintained compiler, runtime, and library compatibility work. Its product goal is to let an operator install a compatibility toolset on a machine with older NVIDIA GPUs and run CUDA 13+ applications there with correct results, as though their required software profile were supported. The intended experience is host-level and broad across applications; each application, API, kernel, and library path still needs an implementation and evidence before it can be claimed. PyTorch CUDA 13 is the first target, not the product boundary. The current prototype is only an opt-in per-process Python launcher and is not yet a machine-wide installer. The first reference hardware is three GeForce GTX TITAN X (GM200, compute capability 5.2) cards. Success means documented workloads build, execute, and pass their own correctness checks on that hardware, with native, translated, or emulated behavior identified and measured. A version string, successful import, or GPU-name spoof alone does not count as compatibility.

The project aims to recover useful semantics of newer CUDA features on older hardware, including features that NVIDIA no longer compiles or supports for that hardware. Age-based support cutoffs alone are not grounds to exclude a feature: evaluate source translation, runtime adaptation, library replacement, or software emulation. When behavior depends on a physical capability, implement a correct software path when feasible and report its resource and performance costs. Claim only behavior that is implemented and validated; keep an explicit evidence-backed boundary for behavior that cannot yet be reproduced.

## Desired state

```text
Modern CUDA software rejects older GPUs or their toolkit
                         ↓
Inspect requirements, compile/translate features, adapt APIs,
and provide software implementations where needed
                         ↓
More tested workloads produce correct results on older NVIDIA hardware
```

## Constraints

- Keep actual device identity and compute capability visible; only virtualize a capability when its behavior is implemented and tested.
- Do not modify the reference host's GPU driver, kernel modules, boot settings, or system CUDA installation as part of routine experiments.
- Do not redistribute NVIDIA proprietary drivers, toolkit files, or libraries. Users supply any required NVIDIA components under their own applicable terms.
- Project code is dual-licensed under MIT OR Apache-2.0; retain both license texts in source and release artifacts.
- Preserve numerical and synchronization semantics before claiming compatibility; report performance separately from correctness.
- Treat each framework, library, API, and GPU architecture as an explicit compatibility target. Unknown requirements fail closed.
- Initial implementation uses Python for host-side inspection, profiles, and framework adapters; device compilation and kernels use the appropriate CUDA C++/PTX/LLVM toolchain. Rust remains an option for a measured runtime component.

## Non-goals

- Claiming universal compatibility for every CUDA 13+ application or arbitrary precompiled binary without implementation and workload-specific evidence. Rebuild, translation, library replacement, or emulation paths should be investigated where technically and legally available.
- Creating unsupported physical instructions or matching newer GPU performance by assertion.
- Claiming complete CUDA API coverage before each behavior is implemented and validated.
- Installing or modifying proprietary host drivers without explicit authorization.
- Creating a public GitHub repository, publishing binaries, or claiming vendor affiliation in this local project phase.

## Success questions

- Which exact software requirement is a toolkit-version check, and which requires newer hardware or library kernels?
- Can the supported workload be rebuilt or translated for the actual compute capability?
- Which APIs and library routines are native, adapted, replaced, or emulated?
- Does the output match an independent reference, and what throughput/latency does it achieve across one and multiple cards?
- Can another operator reproduce the result using only documented dependencies and commands?

## Termination / dormancy

Retire a compatibility target when its software cannot be made correct within available hardware, licensing, or maintenance constraints. Preserve the evidence and failure reason. Keep the broader project active only while a tested workload or a credible, bounded experiment remains.
