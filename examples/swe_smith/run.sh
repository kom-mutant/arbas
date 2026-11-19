#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "$PWD/.venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source "$PWD/.venv/bin/activate"
fi

has_gpu_allocation="${CUDA_VISIBLE_DEVICES+x}"
allocated_devices="${CUDA_VISIBLE_DEVICES:-}"

profile="machine"
runner=""
if [ "${1:-}" = controller ]; then
  if [ "${2:-}" = local ] || [ "${2:-}" = k8s ]; then
    runner="$2"
    if [ -n "${3:-}" ] && [[ "${3:-}" != -* ]]; then
      case "$3" in
        *[!a-z0-9-]*) echo "ERROR: invalid machine name." >&2; exit 2 ;;
      esac
      profile="$3"
    fi
  elif [ -n "${2:-}" ] && [[ "${2:-}" != -* ]]; then
    case "$2" in
      *[!a-z0-9-]*) echo "ERROR: invalid machine name." >&2; exit 2 ;;
    esac
    profile="$2"
  fi
fi

if [ -n "$runner" ]; then
  export AGL_RUNNER="$runner"
fi

if [ -n "${SLURM_CACHE_DIR:-}" ]; then
  export HF_HOME="$SLURM_CACHE_DIR/huggingface"
fi

env_file="${ARBAS_MACHINE_FILE:-$HOME/.config/arbas/$profile.env}"
if [ -f "$env_file" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$env_file"
  set +a
elif [ "$profile" != machine ] && [ "$profile" != local ] && [ "$profile" != k8s ] || [ -n "${ARBAS_MACHINE_FILE:-}" ]; then
  echo "ERROR: machine settings not found at $env_file." >&2
  exit 1
fi

if [ -n "$has_gpu_allocation" ]; then
  export CUDA_VISIBLE_DEVICES="$allocated_devices"
fi

umask 077
exec python -u -m examples.swe_smith.launch "$@"
