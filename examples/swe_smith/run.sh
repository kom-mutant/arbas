#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -x "$PWD/.venv/bin/python" ]; then
  export VIRTUAL_ENV="$PWD/.venv"
  export PATH="$PWD/.venv/bin:$PATH"
fi

umask 077
exec python -u -m examples.swe_smith.launch "$@"
