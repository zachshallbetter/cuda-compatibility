#!/bin/sh
# Install cuda-legacy (CUDA Compatibility) into the Python environment that holds your
# PyTorch CUDA wheel, then run its guided setup.
#
#   curl -fsSL https://cudacompat.com/install.sh | sh
#   curl -fsSL https://cudacompat.com/install.sh | sh -s -- --dry-run
#
# Read it first: https://cudacompat.com/install.sh, checksum at install.sh.sha256,
# source at https://github.com/zachshallbetter/cuda-compatibility/blob/main/site/install.sh
#
# Options:
#   --python PATH   the Python of the environment that holds PyTorch (default: found)
#   --yes           do not ask before downloading PyTorch (when no environment has it) or LLVM
#   --dry-run       print each step without changing anything
#   --uninstall     remove what setup created, then uninstall the package
#   --allow-root    run as root outside a container (a disposable VM, say); never with sudo
#   --version       print the release this script installs
#   -- ARGS         pass the remaining arguments to `cuda-legacy setup`
#
# Needs Linux x86_64, Python 3.10+ and curl or wget. It uses the virtualenv or conda environment
# that holds your PyTorch CUDA wheel; with none, it offers to create ~/torch-cu130 with the
# qualified wheel. Run it as your own user; root is accepted only inside a container
# or with --allow-root. It never uses sudo, and it never changes drivers, the system CUDA
# install or the system Python.
#
# Exit codes: 0 ready; 1 installed, but setup lists what still needs attention; 2 stopped.
#
# Everything runs inside main(), which the last line calls, so a partial download runs nothing.

VERSION=0.2.20
ARCHIVE_URL="https://github.com/zachshallbetter/cuda-compatibility/archive/refs/tags/v$VERSION.tar.gz"
# SHA-256 of ARCHIVE_URL, written by the release. Empty means no release is pinned yet.
ARCHIVE_SHA256=993cdfa3f2f3ee4c5a8b765cb0d419dd68d9ec018be1bd0a64f39208f221c860
PACKAGE=cuda-legacy-compat
# The PyTorch build the adapter is qualified on, and where the installer puts it when you have none (#202).
TORCH_VERSION=2.14.0+cu130
TORCH_INDEX=https://download.pytorch.org/whl/cu130
ENV_MARKER=.cuda-legacy-created
TORCH_HINT="python3 -m venv ~/torch-cu130 && ~/torch-cu130/bin/pip install torch==$TORCH_VERSION --index-url $TORCH_INDEX"

say() { printf '==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
stop() {
    printf 'install stopped: %s\n' "$1" >&2
    shift
    for line in "$@"; do printf '    %s\n' "$line" >&2; done
    exit 2
}

# Print a command; run it unless this is a dry run.
step() {
    printf '    +'
    printf ' %s' "$@"
    printf '\n'
    [ "$DRY_RUN" = 1 ] || "$@"
}

# True inside a Docker or Podman container (CUDA_LEGACY_CONTAINER_MARKERS overrides the files, for tests).
in_container() {
    [ -n "${container:-}" ] && return 0
    for marker in ${CUDA_LEGACY_CONTAINER_MARKERS-/.dockerenv /run/.containerenv}; do
        [ -e "$marker" ] && return 0
    done
    return 1
}

# Refuse root on a workstation, where pip would leave root-owned files in your environment and home.
# Containers usually run as root and are disposable, so root is fine there; sudo never is.
check_user() {
    [ "$(id -u)" = 0 ] || return 0
    [ -z "${SUDO_USER:-}" ] || stop "Run this as yourself, without sudo." \
        "Everything installs into your Python environment and your home directory, so it needs no root."
    if [ "$ALLOW_ROOT" = 1 ]; then
        note "Running as root (--allow-root)."
    elif in_container; then
        note "Running as root inside a container."
    else
        stop "Run this as your own user, not root." \
            "Everything installs into your Python environment and your home directory." \
            "In a disposable machine where root is the only user, pass --allow-root."
    fi
}

usage() {
    cat <<'EOF'
Install cuda-legacy into the Python environment that holds your PyTorch CUDA wheel,
then run its guided setup.

Usage: curl -fsSL https://cudacompat.com/install.sh | sh -s -- [options] [-- setup-args]

  --python PATH   the Python of the environment that holds PyTorch (default: found)
  --yes           do not ask before downloading PyTorch (when no environment has it) or LLVM
  --dry-run       print each step without changing anything
  --uninstall     remove what setup created, then uninstall the package
  --allow-root    run as root outside a container (a disposable VM, say); never with sudo
  --version       print the release this script installs
  -- ARGS         pass the remaining arguments to `cuda-legacy setup`

To upgrade, run the same command again: it replaces the package and keeps the saved setup.
Run it as your own user. Root is accepted inside a container (Docker, Podman) or with --allow-root.

Exit codes: 0 ready; 1 installed, but setup lists what still needs attention; 2 stopped.
EOF
}

# The installed torch version, read from package metadata: importing torch is slow and noisy.
torch_version() {
    "$1" -c 'from importlib.metadata import version; print(version("torch"))' 2>/dev/null
}

# True when the interpreter holds a CUDA build of PyTorch (its version contains "+cu").
has_cuda_torch() {
    case $(torch_version "$1") in
        *+cu*) return 0 ;;
        *) return 1 ;;
    esac
}

# The installed cuda-legacy version, empty when it is not installed.
package_version() {
    "$1" -c 'from importlib.metadata import version; print(version("cuda-legacy-compat"))' 2>/dev/null || true
}

# What this run does to the environment, in words: install, upgrade, reinstall or replace.
change_line() {
    installed=$(package_version "$1")
    if [ -z "$installed" ]; then
        printf 'Installing cuda-legacy %s into %s' "$VERSION" "$1"
    elif [ "$installed" = "$VERSION" ]; then
        printf 'Reinstalling cuda-legacy %s in %s' "$VERSION" "$1"
    elif "$1" -c 'import sys; a, b = (tuple(int(p) for p in v.split(".")) for v in sys.argv[1:]); sys.exit(0 if a < b else 1)' \
        "$installed" "$VERSION" 2>/dev/null; then
        printf 'Upgrading cuda-legacy %s to %s in %s' "$installed" "$VERSION" "$1"
    else
        printf 'Replacing cuda-legacy %s with %s in %s' "$installed" "$VERSION" "$1"
    fi
}

has_package() {
    "$1" -c 'import cuda_legacy_compat' >/dev/null 2>&1
}

# The same order `cuda-legacy setup` uses, so both pick the same environment.
candidates() {
    [ -n "${CUDA_COMPAT_PYTHON:-}" ] && printf '%s\n' "$CUDA_COMPAT_PYTHON"
    [ -n "${VIRTUAL_ENV:-}" ] && printf '%s\n' "$VIRTUAL_ENV/bin/python"
    [ -n "${CONDA_PREFIX:-}" ] && printf '%s\n' "$CONDA_PREFIX/bin/python"
    command -v python3 2>/dev/null || true
}

# find_python TEST: the first candidate that passes TEST (has_cuda_torch or has_package).
find_python() {
    if [ -n "$PYTHON" ]; then
        printf '%s\n' "$PYTHON"
        return 0
    fi
    found=$(candidates | while IFS= read -r candidate; do
        if [ -x "$candidate" ] && "$1" "$candidate"; then
            printf '%s\n' "$candidate"
            break
        fi
    done)
    [ -n "$found" ] || return 1
    printf '%s\n' "$found"
}

check_python() {
    py=$1
    [ -x "$py" ] || stop "$py is not an executable Python."
    "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null ||
        stop "$py is older than Python 3.10." "Use the Python of a newer environment with --python."
    has_cuda_torch "$py" || stop "$py has no PyTorch CUDA wheel." \
        "Install one into a virtual environment, then rerun with --python:" "$TORCH_HINT"
    # A virtualenv or conda environment, never the system Python: pip needs no sudo there.
    "$py" -c 'import os, sys; sys.exit(0 if sys.prefix != sys.base_prefix or os.path.isdir(os.path.join(sys.prefix, "conda-meta")) else 1)' ||
        stop "$py is a system Python, and this installer does not change it." \
            "Install PyTorch into a virtual environment, then rerun with --python:" "$TORCH_HINT"
    prefix=$("$py" -c 'import sys; print(sys.prefix)')
    [ -w "$prefix" ] || stop "You cannot write to $prefix, and this installer does not use sudo." \
        "Use an environment you own."
}

# No environment holds a PyTorch CUDA wheel: offer to create one with the qualified wheel, asking first as for
# LLVM. Sets py to its Python. In a dry run it prints the steps and the rest of the install, and returns 1.
create_torch_env() {
    env=${CUDA_LEGACY_TORCH_ENV:-$HOME/torch-cu130}
    say "No Python with a PyTorch CUDA wheel was found"
    note "Looked at CUDA_COMPAT_PYTHON, the active virtualenv or conda env, and python3 on PATH."
    note "This installer can create one: a virtual environment at $env with torch $TORCH_VERSION"
    note "(the CUDA 13 wheel the adapter is qualified on) and numpy from $TORCH_INDEX:"
    note "about 3 GB to download, 5.3 GB on disk."
    note "To use an environment you already have instead, rerun with --python PATH."
    if [ "$YES" = 0 ]; then
        (: </dev/tty) 2>/dev/null || stop "There is no terminal to ask on before downloading PyTorch." \
            "To create $env, rerun with --yes: curl -fsSL https://cudacompat.com/install.sh | sh -s -- --yes" \
            "To use an environment you have, rerun with: curl -fsSL https://cudacompat.com/install.sh | sh -s -- --python PATH"
        printf '    Create it? [y/N] '
        read -r answer </dev/tty || answer=
        case $answer in
            y | Y | yes | Yes | YES) ;;
            *) stop "Nothing was changed." "Or set up PyTorch yourself, then rerun with --python:" "$TORCH_HINT" ;;
        esac
    fi
    base=$(command -v python3) || stop "python3 was not found." "Install Python 3.10 or newer from your distribution."
    "$base" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null ||
        stop "$base is older than Python 3.10." "Install a newer Python, then rerun with --python, or put it first on PATH."
    if [ -e "$env" ] && [ ! -x "$env/bin/python" ]; then
        stop "$env exists and is not a virtual environment." "Move it aside, or rerun with --python PATH."
    fi
    say "Creating $env"
    if [ ! -x "$env/bin/python" ]; then
        step "$base" -m venv "$env" || stop "python3 could not create a virtual environment at $env." \
            "On Debian and Ubuntu, venv is a separate package: sudo apt install python3-venv" \
            "Then rerun the same command."
    fi
    # Without --quiet: the download takes minutes, and pip shows its progress. numpy too: the wheel doesn't need it,
    # but without it every script's first line is PyTorch's "Failed to initialize NumPy" warning.
    step "$env/bin/python" -m pip install --no-input "torch==$TORCH_VERSION" numpy --index-url "$TORCH_INDEX" ||
        stop "pip could not install torch $TORCH_VERSION into $env." "pip's message is above." \
            "Rerunning the same command resumes into the same environment."
    py=$env/bin/python
    CREATED_ENV=$env
    # --uninstall offers to delete only an environment that carries this marker (#253).
    [ "$DRY_RUN" = 1 ] || printf 'Created by the cuda-legacy installer; --uninstall offers to delete it.\n' > "$env/$ENV_MARKER"
    if [ "$DRY_RUN" = 1 ]; then
        note "Then cuda-legacy $VERSION installs into $env and setup runs there."
        say "Dry run: nothing was changed"
        return 1
    fi
}

scripts_dir() {
    "$1" -c 'import sysconfig; print(sysconfig.get_path("scripts"))'
}

fetch() {
    case $1 in
        https://*) ;;
        *)
            cp "$1" "$2"
            return
            ;;
    esac
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --proto '=https' --tlsv1.2 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --https-only -O "$2" "$1"
    else
        stop "Neither curl nor wget is installed."
    fi
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d ' ' -f 1
    else
        stop "Neither sha256sum nor shasum is installed, so the download cannot be verified."
    fi
}

do_install() {
    say "Checking this machine"
    if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
        stop "cuda-legacy runs on Linux x86_64 with an NVIDIA GPU; this is $(uname -s) $(uname -m)."
    fi
    command -v nvidia-smi >/dev/null 2>&1 ||
        warn "nvidia-smi was not found. Install the NVIDIA driver before running workloads; setup checks it."

    say "Finding the Python environment that holds PyTorch"
    if ! py=$(find_python has_cuda_torch); then
        create_torch_env || return 0
    fi
    check_python "$py"
    note "$py (torch $(torch_version "$py"))"

    if [ "$YES" = 0 ] && ! (: </dev/tty) 2>/dev/null; then
        stop "There is no terminal to ask on before setup downloads LLVM (about 1.2 GB)." \
            "Rerun with --yes: curl -fsSL https://cudacompat.com/install.sh | sh -s -- --yes"
    fi

    url=${CUDA_LEGACY_ARCHIVE:-$ARCHIVE_URL}
    sha=${CUDA_LEGACY_ARCHIVE_SHA256:-$ARCHIVE_SHA256}
    [ -n "$sha" ] || stop "This copy of the installer has no release pinned yet." \
        "Install from a checkout instead: https://cudacompat.com/docs/getting-started.html"

    say "Downloading cuda-legacy $VERSION"
    note "$url"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    trap 'rm -rf "$tmp"; exit 2' HUP INT TERM
    archive="$tmp/cuda-legacy-$VERSION.tar.gz"
    if [ "$DRY_RUN" = 0 ]; then
        fetch "$url" "$archive" || stop "The download failed: $url"
        actual=$(sha256_of "$archive")
        [ "$actual" = "$sha" ] || stop "The download does not match the published checksum." \
            "expected $sha" "received $actual" "Nothing was installed."
        note "SHA-256 verified"
    fi

    say "$(change_line "$py")"
    step "$py" -m pip install --no-input --quiet --upgrade "$archive" ||
        stop "pip could not install $PACKAGE into $py." "pip's message is above. Nothing else was changed."

    bin=$(scripts_dir "$py")/cuda-legacy
    say "Running setup"
    [ "$YES" = 1 ] && set -- --yes "$@"
    if [ "$DRY_RUN" = 1 ]; then
        step "$bin" setup --python "$py" "$@"
        say "Dry run: nothing was changed"
        return 0
    fi
    set +e
    if [ "$YES" = 1 ]; then
        "$bin" setup --python "$py" "$@"
    else
        # Piped from curl, this script's stdin is the download; setup asks on the terminal.
        "$bin" setup --python "$py" "$@" </dev/tty
    fi
    status=$?
    set -e

    case $status in
        0)
            say "Ready"
            if [ -n "$CREATED_ENV" ]; then
                note "Use the new environment in a shell: . $CREATED_ENV/bin/activate"
                note "Then run a script: cuda-legacy run app.py"
            else
                note "Run a script: $bin run app.py"
            fi
            note "Upgrade later: run the same command again"
            note "Remove it all: curl -fsSL https://cudacompat.com/install.sh | sh -s -- --uninstall"
            ;;
        1)
            say "Installed. Setup lists what still needs attention above."
            note "After fixing it, check again: $bin setup"
            ;;
        *)
            say "Installed, but setup stopped. Its message is above."
            note "Rerun it with: $bin setup"
            ;;
    esac
    return "$status"
}

do_uninstall() {
    say "Finding the Python environment that holds $PACKAGE"
    if ! py=$(find_python has_package) || ! has_package "$py"; then
        say "$PACKAGE is not installed in any environment this script can find. Nothing to remove."
        note "If it is elsewhere, rerun with --python PATH."
        return 0
    fi
    note "$py"
    bin=$(scripts_dir "$py")/cuda-legacy
    say "Removing what setup created"
    if [ -x "$bin" ]; then
        step "$bin" setup --remove || warn "setup --remove failed; continuing with the package."
    fi
    prefix=$("$py" -c 'import sys; print(sys.prefix)')
    say "Uninstalling $PACKAGE"
    step "$py" -m pip uninstall --yes --quiet "$PACKAGE"
    # pip leaves the profile data files' directories behind, empty.
    [ "$DRY_RUN" = 1 ] || rmdir "$prefix/share/cuda-legacy-compat/profiles" "$prefix/share/cuda-legacy-compat" 2>/dev/null || true
    if [ -f "$prefix/$ENV_MARKER" ] && remove_created_env "$prefix"; then
        say "Removed, with the PyTorch environment this installer created. Drivers and the system CUDA install were not touched."
    elif [ -f "$prefix/$ENV_MARKER" ]; then
        say "Removed. Kept $prefix, the PyTorch environment this installer created."
        note "Delete it too, if nothing else uses it: rm -rf $prefix"
    else
        say "Removed. Drivers, the system CUDA install and PyTorch were not touched."
    fi
}

# The environment the installer created (#253): its size, and on a yes typed at the terminal, delete it. Never by
# --yes alone, and never a path that could be more than that environment.
remove_created_env() {
    env=$1
    case $env in
        /*) ;;
        *) return 1 ;;
    esac
    [ "$env" != / ] && [ "$env" != "$HOME" ] && [ -x "$env/bin/python" ] || return 1
    size=$(du -sh "$env" 2>/dev/null | cut -f 1)
    note "This installer created the PyTorch environment $env (${size:-its size unknown})."
    (: </dev/tty) 2>/dev/null || return 1
    printf '    Delete it too? [y/N] '
    read -r answer </dev/tty || answer=
    case $answer in
        y | Y | yes | Yes | YES) ;;
        *) return 1 ;;
    esac
    step rm -rf "$env"
}

main() {
    set -eu
    # pip's build environment is a child process, so these reach it where flags would not:
    # the install leaves no cache or version-check file behind.
    PIP_NO_CACHE_DIR=1
    PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_NO_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK
    PYTHON=
    CREATED_ENV=
    ALLOW_ROOT=0
    YES=0
    DRY_RUN=0
    ACTION=do_install
    while [ $# -gt 0 ]; do
        case $1 in
            --python)
                [ $# -ge 2 ] || stop "--python needs a path."
                PYTHON=$2
                shift
                ;;
            --python=*) PYTHON=${1#--python=} ;;
            -y | --yes) YES=1 ;;
            -n | --dry-run) DRY_RUN=1 ;;
            --uninstall) ACTION=do_uninstall ;;
            --allow-root) ALLOW_ROOT=1 ;;
            --version)
                printf 'cuda-legacy installer %s\n' "$VERSION"
                return 0
                ;;
            -h | --help)
                usage
                return 0
                ;;
            --)
                shift
                break
                ;;
            *) stop "Unknown option: $1" "Run with --help for the options." ;;
        esac
        shift
    done

    check_user
    "$ACTION" "$@"
}

main "$@"
