# Decision 0001: Python host tooling; source-built workload compatibility first

**Status:** Accepted for initial implementation<br>
**Date:** 2026-10-31

## Decision

Implement host inspection, compatibility-profile auditing, and initial framework adapters in Python. Keep device work in CUDA C++/PTX and evaluate LLVM/Clang as an independently controlled source frontend. First target source-built or rebuildable workloads; defer arbitrary binary interception.

## Why

The first high-value users are PyTorch and similar Python framework users. Python makes it practical to inspect wheel metadata, dispatch supported adapters, and test application-level behavior. LLVM and GPU code already require native compiler tooling, so a Rust host rewrite would not remove that dependency. The verified baseline uses a small CUDA C++ kernel and is kept as an independent execution test.

## Consequences

- The Python core starts with no third-party runtime dependencies.
- CLI and profile behavior can be validated on systems without NVIDIA hardware.
- GPU-specific compilation, runtime calls, and numerical kernels remain native components with separate tests.
- Rust remains available for a measured component; no Rust runtime is introduced in the first vertical slice.
- Current and future capability reports remain explicit; metadata spoofing is not used to mask absent implementations.

## Revisit when

Move an identified component to Rust only if the current Python/native boundary causes a demonstrated correctness, safety, performance, or packaging problem.
