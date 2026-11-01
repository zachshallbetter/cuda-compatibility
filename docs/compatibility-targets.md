# Compatibility targets

Evidence snapshot: 2026-10-31. This matrix tracks framework/toolkit packaging separately from the hardware capability an application actually needs.

| Target | Newer CUDA dependency | Hardware / package boundary | Initial disposition |
| --- | --- | --- | --- |
| PyTorch 2.14.0 CUDA 13.0.3 build | Official release matrix lists CUDA 13.0.3 and CUDA 13.2.1 builds | Its CUDA 13 architecture matrix starts at Turing (`sm_75`) | First compatibility target; selected workloads must be rebuilt or adapted for Maxwell, and each API/kernel path needs evidence |
| PyTorch 2.14.0 CUDA 12.6.3 build | Official release matrix lists CUDA 12.6.3 | Its architecture matrix includes Maxwell (`sm_50`) | Maxwell-supported comparison baseline; profile does not imply every operator or dependency works on GM200 |
| TensorRT 11.4.1 | CUDA 13.4 Update 1 package; matching CUDA 12.x pip wheels also exist | Minimum supported GPU is `sm_75`; Maxwell is outside support | Useful negative/control case for a strong hardware gate; not a first compatibility target |
| vLLM current source | CUDA 12.8, 13.0, and PyTorch release wheels are documented | Current source and kernel capability needs require separate version-pinned audit before adopting as a target | Candidate only after a specific workload and current minimum architecture are captured |

Do not mark a project “CUDA 13 required” when it offers another toolkit build. Do not infer a newer CUDA toolkit requirement from `nvidia-smi`'s “CUDA Version” field; it reports driver capability, not the installed toolkit. For a prebuilt wheel, toolkit version, bundled CUDA runtime, embedded GPU architectures, driver minimum, and library kernels are distinct evidence items.

## Source records

- [PyTorch 2.14.0 release and CUDA architecture compatibility matrix](https://github.com/pytorch/pytorch/blob/v2.14.0/RELEASE.md) — release 2.14 rows list CUDA 12.6.3 / 13.0.3 / 13.2.1; published CUDA 13 architecture set starts at `sm_75`, while CUDA 12.6.3 includes Maxwell (`sm_50`).
- [TensorRT 11.4.1 support matrix](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/getting-started/support-matrix.html) — `sm_75` minimum; Linux x86-64 packages include CUDA 13.4 Update 1 and matching 12.x wheels.
- [TensorRT architecture overview](https://docs.nvidia.com/deeplearning/tensorrt/latest/architecture/architecture-overview.html) — TensorRT 8.5.3 was last supporting Maxwell; TensorRT 11.4.1 requires `sm_75+`.
- [vLLM CUDA install guide](https://github.com/vllm-project/vllm/blob/main/docs/getting_started/installation/gpu.cuda.inc.md) — current wheel options; exact compute-capability claim is intentionally deferred pending a version-specific source and tested workload.
- [CUDA Toolkit 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html) — Maxwell/Pascal/Volta offline compilation and library support removed in CUDA 13.
- [CUDA 12.9 nvcc guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html) — `compute_52` is in the Maxwell target family.
