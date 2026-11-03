# CUDA Compatibility

**Run supported newer CUDA workloads on older NVIDIA GPUs.**

**Project site:** [CUDA Compatibility](https://cudacompat.com/) · [Zach Shallbetter](https://zachshallbetter.com/) · [GitHub profile](https://github.com/zachshallbetter)

The static project site is in `site/` and is served at https://cudacompat.com/ by the Vercel project `cuda-compatibility` (`www.cudacompat.com` redirects there), whose Root Directory is `site` and which has no build step. Pull requests run the project checks and test suite and receive a Vercel preview deployment; merging to `main` deploys production. Fonts, scripts and the hero wireframe are served from `site/` itself, so a page load makes no third-party request before analytics consent.

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

For development from a source checkout, use `python -m pip install -e .` instead. To use `run`, install this package into the **same Python environment** that contains the target PyTorch CUDA 13 wheel; the launcher starts the script with that environment's `sys.executable` unless `--python` (or `CUDA_COMPAT_PYTHON`) names another interpreter.

On Linux, Homebrew can install the CLI in its own environment. The formula installs the tagged [0.1.0 release](https://github.com/zachshallbetter/cuda-compatibility/releases/tag/v0.1.0); add `--HEAD` to build `main`:

```sh
brew tap zachshallbetter/cuda-compatibility https://github.com/zachshallbetter/cuda-compatibility
brew install cuda-legacy
export CUDA_COMPAT_PYTHON=/path/to/your-env/bin/python   # the environment that holds PyTorch
```

The formula is in `Formula/cuda-legacy.rb`; its install has not yet been run on the reference host. There is no standalone binary.

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

The launcher compiles selected kernels to real `sm_52` PTX, builds a driver bridge and runtime-version facade in a temporary directory, then enables the adapter only in the child process. It removes its generated build directory when the script exits. The facade reports CUDA Runtime 13.0 for that process while retaining actual GPU identity and compute capability. The adapter includes selected native PyTorch kernels and CPU fallbacks capped at 64 MiB of inputs and outputs per operation. First-order gradients and SGD are experimentally verified only for a small float32 linear→ReLU→sum workload; this is not general training, PyTorch, or CUDA 13 support. Because the launcher uses `LD_PRELOAD` and Python startup customization, run only scripts you trust.

From a source checkout, a profile can be named by path (`profiles/pytorch-2.14-cu130.json`); an installed package also accepts the profile filename or ID (`pytorch-2.14-cu130`).

For the complete install-to-run workflow, diagnostic meanings, exit codes, and troubleshooting steps, see the [user guide](docs/user-guide.md).

`cuda-legacy` is the short command. `cuda-legacy-compat` remains installed as a backward-compatible alias. Install the launcher into the same Python environment as the CUDA-enabled PyTorch wheel; see the user guide for the virtual-environment and user-site paths.

## Licensing

Project code is dual-licensed under **MIT OR Apache-2.0**. NVIDIA drivers, headers, toolkit binaries, runtime libraries, and other proprietary components are not bundled.

## Current status

- **Verified:** Native CUDA 12.4 `sm_52` vector-add kernel passed on all three reference GPUs; see [Maxwell baseline](docs/research/maxwell-baseline.md).
- **Experimental:** LLVM 23.1.4 compiled PTX 8.4 for `sm_52`, and a process-local shim returned CUDA Runtime 13.0.0 while preserving real device properties. The official PyTorch 2.14 CUDA 13 wheel passed an operator/MLP suite on all three cards. On broker-selected GPU 2, one owner-supplied transformer checkpoint matched CPU output across six tested batch/sequence configurations; a separate convolution classifier used CPU fallback for convolution. New float16 and strided matrix cases passed on GPUs 1 and 2, as did first-order gradients and seven `foreach=False` SGD updates for a 2×2 linear→ReLU→sum workload. The tiny training probe measured 2.360 ms on GPU 2 and 7.296 ms on GPU 1 median end-to-end latency, with 67,180,032 bytes peak allocated on each; this is diagnostic, not representative performance. Broader model, optimizer, library, and per-card coverage remains open. See [PyTorch CUDA 13 on Maxwell](docs/research/pytorch-cu130-maxwell.md).
- **Try it:** [`examples/sm52-mlp`](examples/sm52-mlp/README.md) pins one workload to a release, machine and expected output, shows which route each operation takes, and links a [reproduction report](https://github.com/zachshallbetter/cuda-compatibility/issues/new?template=reproduction-report.yml) form.
- **Implemented:** Read-only inventory and profile auditing. The target host is incompatible with the published PyTorch CUDA 13 build profile (`sm_75` minimum) and meets the published CUDA 12.6 Maxwell architecture floor (`sm_50`); the latter is a package-architecture check, not an end-to-end operator certification.
- **Not yet implemented:** broad CUDA runtime mappings, general PyTorch operator/library coverage, arbitrary autograd/training compatibility, independently assembled CUDA execution, or multi-GPU workload scheduling.
- **Not claimed:** General CUDA 13 binary compatibility or equal performance to newer cards.

## Project map

- [`src/cuda_legacy_compat/cli.py`](src/cuda_legacy_compat/cli.py) — host inspection, profile auditing, diagnostics, and CLI entry point.
- [`src/cuda_legacy_compat/launch.py`](src/cuda_legacy_compat/launch.py) — temporary build and process-scoped launcher.
- [`src/cuda_legacy_compat/torch_adapter.py`](src/cuda_legacy_compat/torch_adapter.py) — registered PyTorch operator routes and bounded fallbacks.
- [`src/cuda_legacy_compat/compat/`](src/cuda_legacy_compat/compat/) — packaged kernels, driver bridge, runtime facade, and opt-in startup hook.
- [`profiles/`](profiles/) — versioned, source-backed framework/tool requirements.
- [`experiments/maxwell/`](experiments/maxwell/) — native kernel baseline and reproducible runner.
- [`experiments/pytorch-cu130/`](experiments/pytorch-cu130/) — stock-wheel baseline and end-to-end launcher probes.
- [`examples/sm52-mlp/`](examples/sm52-mlp/) — public example with pinned configuration, expected output and execution routes.
- [`docs/research/`](docs/research/) — sourced technical findings and experiment records.
- [`docs/decisions/`](docs/decisions/) — architecture and language decisions.
- [`docs/WORK_ITEMS.md`](docs/WORK_ITEMS.md) — bounded local work frontier until a remote provider is bound.

## Development

```sh
make check
make test
python3 .agents/bootstrap/scripts/gen-context.py --check
python3 .agents/bootstrap/scripts/validate-bootstrap.py --check-context
```

See [the architecture](docs/architecture.md), the [implementation inventory](docs/architecture/implementation-inventory.md) (what is implemented, experimental or proposed, and how open issues map onto it), [target profiles](docs/compatibility-targets.md), and [roadmap](docs/roadmap.md). Project authority and operating rules are in `.agents/project/intent.md` and `AGENTS.md`.
