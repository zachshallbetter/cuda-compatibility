# Security policy

CUDA Compatibility interposes on a Python process that runs PyTorch, so a flaw can matter even though the project is experimental.

## Reporting a vulnerability

Report privately through [GitHub security advisories](https://github.com/zachshallbetter/cuda-compatibility/security/advisories/new). Please do not open a public issue for a suspected vulnerability.

Include the version (`cuda-legacy --version`), the command you ran, the GPU and driver, and what an attacker could do.

## Scope

- The `cuda-legacy` launcher, its runtime facade, native kernels, and CPU fallback routes.
- The site at https://cudacompat.com/.

`cuda-legacy` runs the script you give it with full user privileges. Running an untrusted script through it is not a vulnerability in this project.

## Supported versions

Only the latest release receives fixes.
