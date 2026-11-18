# Contributing to CUDA Compatibility

Thanks for your interest. This project is experimental and maintained by one person; small, focused changes with evidence are easiest to review.

Project authority and operating rules are in [`AGENTS.md`](AGENTS.md) and `.agents/project/intent.md`. Read `AGENTS.md` before changing anything under `.agents/`.

## Setup

Python 3.10 or newer. The package has no runtime dependencies, and the checks run from a clone without installing it.

```sh
make check    # bootstrap validation, protocol integrity, site checks, site metadata
make test     # the unit tests (unittest discover over tests/)
make smoke    # install into a throwaway venv and drive the first-run path (no GPU needed)
```

`make context` regenerates `.agents/context-lock.json` and the compiled context after you edit a file it includes; `make check` fails until it is current. CI also runs `shellcheck -s sh site/install.sh`.

Changes to the site follow [the site protocol](docs/SITE_PROTOCOL.md): `make site` regenerates and checks.

## Conventions

- **Conventional commits:** `docs: ...`, `ci: ...`, `chore: ...`, `fix(scope): ...`, `docs(rules): ...`. The merge adds the PR number.
- **State only what the repository supports.** Do not claim a GPU, driver or framework combination works without a recorded run. Failed and partial runs are as useful as passing ones.
- **Do not attach NVIDIA files** to issues or pull requests.

## Pull requests

- Branch off `main` and keep the change focused.
- Describe the change and how you verified it. The PR template lists the checks.
- Green CI is required to merge.
