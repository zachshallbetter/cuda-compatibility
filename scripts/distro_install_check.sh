#!/bin/sh
# Run the pinned installer in this Linux distribution, without a GPU, and check what a user would see:
# the installer finds the virtualenv, installs the release, setup names what still needs attention, the
# everyday commands work, and --uninstall leaves nothing of cuda-legacy behind.
#
#   sh scripts/distro_install_check.sh PYTHON
#
# PYTHON is the distribution's python3. Run from a checkout; the installer is site/install.sh, which downloads
# the release archive it pins. Needs network access. Used by .github/workflows/installer-distros.yml.
set -eu

python=${1:?usage: distro_install_check.sh PYTHON}
root=$(cd "$(dirname "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

export HOME="$work/home" XDG_CONFIG_HOME='' XDG_DATA_HOME='' PIP_CACHE_DIR="$work/pip-cache"
mkdir -p "$HOME"
distro=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '"')

echo "== $distro, $("$python" --version)"
"$python" -m venv "$work/venv"
"$work/venv/bin/python" -m pip install --quiet --no-input torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
export VIRTUAL_ENV="$work/venv" PATH="$work/venv/bin:$PATH"

echo "== install (found through VIRTUAL_ENV)"
set +e
sh "$root/site/install.sh" --yes </dev/null >"$work/install.txt" 2>&1
code=$?
set -e
cat "$work/install.txt"
# No GPU and no CUDA headers here, so setup lists what still needs attention (exit 1); 2 means it stopped.
[ "$code" -eq 1 ] || [ "$code" -eq 0 ] || fail "installer exited $code"
grep -q "$work/venv/bin/python" "$work/install.txt" || fail "installer did not use the active virtualenv"

version=$(sed -n 's/^VERSION=//p' "$root/site/install.sh")
[ "$(cuda-legacy --version)" = "cuda-legacy $version" ] || fail "installed version is not $version"

echo "== everyday commands"
cuda-legacy >"$work/bare.txt" 2>&1 || true
grep -q "CUDA Compatibility $version" "$work/bare.txt" || fail "bare cuda-legacy did not print its status"
cuda-legacy ops | grep -q "Runs natively on sm_52" || fail "cuda-legacy ops did not list routes"
set +e
cuda-legacy doctor >"$work/doctor.txt" 2>&1
code=$?
set -e
[ "$code" -eq 1 ] || fail "doctor without a GPU should exit 1 (got $code)"

echo "== uninstall"
sh "$root/site/install.sh" --uninstall --yes </dev/null
if "$work/venv/bin/python" -m pip show cuda-legacy-compat >/dev/null 2>&1; then
    fail "package still installed"
fi
left=$(find "$HOME" -path "*cuda-legacy*")
[ -z "$left" ] || fail "uninstall left: $left"
echo "OK $distro: install, status, ops, doctor and uninstall"
