# CUDA Compatibility

**Run supported newer CUDA workloads on older NVIDIA GPUs.**

**Project site:** [CUDA Compatibility](site/index.html) · [Zach Shallbetter](https://zachshallbetter.com/) · [GitHub profile](https://github.com/zachshallbetter)

The responsive project site is in `site/`. After the project is hosted on GitHub and Pages is configured to use GitHub Actions, a repository maintainer can publish it from **Actions → Publish GitHub Pages → Run workflow**. The workflow uploads only `site/` and runs only when manually dispatched.

This independent research and implementation project targets software whose toolkit, compiler, or library requirements have moved beyond older cards. The end goal is to install a compatibility toolset on a host and run CUDA 13+ applications on its older GPUs with correct results. PyTorch CUDA 13 is the first target. The current prototype is an opt-in, per-process Python launcher; it is not yet a machine-wide installer and does not make arbitrary CUDA 13 applications compatible.

CUDA is NVIDIA's trademark. CUDA Compatibility is an independent project and is not affiliated with NVIDIA.

The first hardware reference is three GTX TITAN X (GM200, `sm_52`) cards on the NVIDIA 580.178.04 driver. A CUDA 12.4 vector-add baseline passed on each GPU. The CLI provides read-only inventory, profile browsing/auditing, launcher diagnostics, and an opt-in PyTorch CUDA 13 launcher whose operator coverage is still experimental.

## Install

Install the package into a Python environment you control. No NVIDIA files are bundled.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

For development from a source checkout, use `python -m pip install -e .` instead. To use `run`, install this package into the **same Python environment** that contains the target PyTorch CUDA 13 wheel; the launcher starts the script with that environment's `sys.executable`. The repository does not currently provide a system-wide installer or a standalone binary.

The `inspect` and `audit` commands require Python 3.10 or newer and no GPU. `run` has only been tested with this configuration:

| Component | Tested configuration |
|---|---|
| OS / Python | Linux x86_64 / Python 3.10.12 |
| PyTorch | Official `torch 2.14.0+cu130`; CUDA Runtime 13.0.96 |
| GPU / driver | GTX TITAN X GM200, `sm_52`; NVIDIA driver 580.178.04 |
| Compiler | LLVM/Clang 23.1.4 with NVPTX support |
| Host development files | CUDA 12.4 headers (`cuda.h`, `cuda_runtime.h`), driver library/linker access, `g++` |

Other configurations may work, but are not yet validated. The adapter requires every visible GPU to report `sm_52`. It uses the host driver and CUDA 12.4 development interface to build and load supported kernels; installing the Python package does not install or change those system components.

## CLI

```sh
cuda-legacy --help
cuda-legacy --version
cuda-legacy inspect
cuda-legacy profiles list
cuda-legacy profiles show pytorch-2.14-cu130
cuda-legacy doctor --llvm-root /path/to/LLVM-23.1.4 --json
cuda-legacy audit profiles/pytorch-2.14-cu130.json
cuda-legacy help run
```

`inspect` prints local GPU/compiler inventory as JSON. `profiles` lists or explains the version-pinned targets. `doctor` checks the prerequisites for the current experimental launcher; `ready` means those prerequisites were found, not that a workload is supported. `audit` compares inventory with a target profile. Its `compatible` result covers only the profile's recorded toolkit and compute-capability checks, not operator, library, driver, or workload execution. Audit exits 0 for compatible, 1 for incompatible, 2 for an invalid profile, and 3 when compatibility is unknown. `doctor` exits 0 when all checked launcher prerequisites are present and 1 otherwise.

The experimental `run` command starts one Python script with the opt-in PyTorch adapter. Set `CUDA_COMPAT_LLVM_ROOT` or pass `--llvm-root`; pass `--cuda-root` if the CUDA 12.4 development files are elsewhere:

```sh
cuda-legacy run \
  --llvm-root /path/to/LLVM-23.1.4 \
  --cuda-root /usr/local/cuda-12.4 \
  -- app.py --your-application-argument
```

The launcher compiles selected kernels to real `sm_52` PTX, builds a driver bridge and runtime-version facade in a temporary directory, then enables the adapter only in the child process. It removes its generated build directory when the script exits. The facade reports CUDA Runtime 13.0 for that process while retaining actual GPU identity and compute capability. The adapter includes selected native PyTorch kernels and inference-only CPU fallbacks capped at 64 MiB per operation. It has been exercised on the workloads below; this is not general PyTorch or CUDA 13 support. Because the launcher uses `LD_PRELOAD` and Python startup customization, run only scripts you trust.

From a source checkout, a profile can be named by path (`profiles/pytorch-2.14-cu130.json`); an installed package also accepts the profile filename or ID (`pytorch-2.14-cu130`).

For the complete install-to-run workflow, diagnostic meanings, exit codes, and troubleshooting steps, see the [user guide](docs/user-guide.md).

`cuda-legacy` is the short command. `cuda-legacy-compat` remains installed as a backward-compatible alias. Install the launcher into the same Python environment as the CUDA-enabled PyTorch wheel; see the user guide for the virtual-environment and user-site paths.

## Licensing

Project code is dual-licensed under **MIT OR Apache-2.0**. NVIDIA drivers, headers, toolkit binaries, runtime libraries, and other proprietary components are not bundled.

## Current status

- **Verified:** Native CUDA 12.4 `sm_52` vector-add kernel passed on all three reference GPUs; see [Maxwell baseline](docs/research/maxwell-baseline.md).
- **Experimental:** LLVM 23.1.4 compiled PTX 8.4 for `sm_52`, and a process-local shim returned CUDA Runtime 13.0.0 while preserving real device properties. The official PyTorch 2.14 CUDA 13 wheel passed an operator/MLP suite on all three cards. On broker-selected GPU 2, one owner-supplied transformer checkpoint matched CPU output across six tested batch/sequence configurations. A separate `2×3×32×32` convolution/pooling classifier also matched CPU output, using CPU fallback for convolution. These workloads combine native Maxwell kernels with bounded CPU fallbacks; they do not establish general PyTorch support. Training, autograd, broader model coverage, and model validation on GPUs 0 and 1 remain open. See [PyTorch CUDA 13 on Maxwell](docs/research/pytorch-cu130-maxwell.md).
- **Implemented:** Read-only inventory and profile auditing. The target host is incompatible with the published PyTorch CUDA 13 build profile (`sm_75` minimum) and meets the published CUDA 12.6 Maxwell architecture floor (`sm_50`); the latter is a package-architecture check, not an end-to-end operator certification.
- **Not yet implemented:** broad CUDA runtime mappings, general PyTorch operator/library coverage, autograd/training compatibility, independently assembled CUDA execution, or multi-GPU workload scheduling.
- **Not claimed:** General CUDA 13 binary compatibility or equal performance to newer cards.

## Project map

- `src/cuda_legacy_compat/` — host inspection and compatibility-profile CLI.
- `src/cuda_legacy_compat/compat/` — packaged adapter kernels, driver bridge, runtime facade, and opt-in startup hook.
- `profiles/` — versioned, source-backed framework/tool requirements.
- `experiments/maxwell/` — native kernel baseline and reproducible runner.
- `experiments/pytorch-cu130/` — stock-wheel baseline and end-to-end launcher probes.
- `docs/research/` — sourced technical findings and experiment records.
- `docs/decisions/` — architecture and language decisions.
- `docs/WORK_ITEMS.md` — bounded local work frontier until a remote provider is bound.

## Development

```sh
make check
make test
python3 scripts/gen-context.py --check
python3 scripts/validate-bootstrap.py --check-context
```

See [the architecture](docs/architecture.md), [target profiles](docs/compatibility-targets.md), and [roadmap](docs/roadmap.md). Project authority and operating rules are in `PROJECT_INTENT.md` and `AGENTS.md`.
