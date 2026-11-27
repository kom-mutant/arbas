#!/usr/bin/env bash
set -euo pipefail

if [ "$(uname -sm)" != "Linux x86_64" ]; then
  echo 'The training wheels require Linux x86_64 with CUDA 12.9 support.' >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

uv venv --managed-python --python 3.12 --allow-existing .venv
python=.venv/bin/python

python_include=$("$python" -c 'import sysconfig; print(sysconfig.get_path("include"))')
if [ ! -r "$python_include/Python.h" ]; then
  echo "Python headers required by Triton are missing: $python_include/Python.h" >&2
  exit 1
fi

patch_hash=$(sha256sum "$SCRIPT_DIR/verl.patch" | cut -c1-12)
wheel_cache="${XDG_CACHE_HOME:-$HOME/.cache}/arbas/wheels/verl-0.8.0-${patch_hash}"

if ! compgen -G "$wheel_cache/verl-*.whl" >/dev/null; then
  build_dir=$(mktemp -d)
  trap 'rm -rf -- "$build_dir"' EXIT

  curl -fsSL https://files.pythonhosted.org/packages/3b/f7/d5ff07531e10f0278d3f2f5bf9a0a9ddd5d8d3d3eb903e2dd34c70bee28f/verl-0.8.0.tar.gz \
    -o "$build_dir/verl-0.8.0.tar.gz"
  (cd "$build_dir" && sha256sum -c - <<'SHA256'
814352ab55179dab0724091313e0367e866af667f6a749cc248b73699eb15a21  verl-0.8.0.tar.gz
SHA256
  )
  tar -xzf "$build_dir/verl-0.8.0.tar.gz" -C "$build_dir"
  patch -d "$build_dir/verl-0.8.0" -p1 < "$SCRIPT_DIR/verl.patch"
  mkdir -p "$build_dir/out" "$(dirname "$wheel_cache")"
  uv build --quiet --python "$python" --wheel --out-dir "$build_dir/out" "$build_dir/verl-0.8.0"
  rm -rf -- "$wheel_cache"
  mv "$build_dir/out" "$wheel_cache"
fi

if [ "${1:-}" = --lock ]; then
  uv --config-file "$SCRIPT_DIR/training-uv.toml" pip compile --python "$python" --torch-backend cu129 \
    --find-links "$wheel_cache" --no-annotate \
    --custom-compile-command 'bash examples/swe_smith/setup_training.sh --lock' \
    --output-file "$SCRIPT_DIR/requirements.txt" "$SCRIPT_DIR/requirements.in"
else
  uv --config-file "$SCRIPT_DIR/training-uv.toml" pip sync --python "$python" --torch-backend cu129 \
    --find-links "$wheel_cache" "$SCRIPT_DIR/requirements.txt"
  uv pip check --python "$python"
fi
