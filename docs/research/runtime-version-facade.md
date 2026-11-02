# Runtime-version facade experiment

**Evidence status:** Reproduced on the reference host, 2026-10-31.<br>
**Scope:** Opt-in interposition of one CUDA Runtime API query; CUDA 12.4.131 and R580 remain the backend.

## Behavior

[`cuda_runtime_version_facade.cpp`](../../src/cuda_legacy_compat/compat/cuda_runtime_version_facade.cpp) interposes `cudaRuntimeGetVersion` for a process using `LD_PRELOAD`. With `CUDA_COMPAT_RUNTIME_VERSION=13000`, that call returns the CUDA 13.0.0 version integer. Only the exact supported value is accepted. With the variable unset, the shim calls the real function through `RTLD_NEXT`. No other CUDA API is interposed.

The probe confirms that CUDA Driver API version reporting, device name, and physical compute capability remain from the installed NVIDIA stack. The tested device reports `NVIDIA GeForce GTX TITAN X`, compute capability 5.2; `driver=13000` is the CUDA Driver API compatibility integer (13.0), not the NVIDIA driver release number. The installed NVIDIA driver release is R580 / 580.178.04.

## Reproduction

On the reference host:

```sh
bash experiments/runtime-facade/run_probe.sh
```

The runner creates a temporary shared shim and links its probe and vector-add workload against the installed CUDA runtime with `--cudart=shared`. It tests the real runtime report, the opt-in virtual version, and the native vector-add workload under the preload on each card. The shim never replaces a system library; all build outputs are removed on exit. On this host, NVCC's default linkage produced no dynamic `libcudart` dependency (`ldd` showed no `cudart` entry), and the preload did not override that probe. This interposition method therefore requires a dynamically linked application; statically linked applications need another approach such as rebuild-time adaptation or binary instrumentation.

## Observed result

For each of the three physical cards:

```text
baseline runtime=12040 driver=13000 device=NVIDIA GeForce GTX TITAN X compute=5.2
facade runtime=13000 driver=13000 device=NVIDIA GeForce GTX TITAN X compute=5.2
PASS device=0 elements=1048576 result=1048576
```

Here `runtime` and `driver` are CUDA API version integers; the driver package release is reported separately as 580.178.04.

The probe also rejected `CUDA_COMPAT_RUNTIME_VERSION=13001`. The compatibility profile is process-local and opt-in.

## Limits

This proves only that one dynamically linked application's runtime-version check can be changed while ordinary calls in the tested CUDA 12.4 workload continue to use the real runtime. It does not add CUDA 13 APIs, device instructions, library kernels, or support to the prebuilt PyTorch CUDA 13 package. Code paths selected after the version query may still fail. Applications using a statically linked runtime, alternate version-query paths, or other loader arrangements are outside this experiment. Do not represent this probe as general CUDA 13 compatibility.
