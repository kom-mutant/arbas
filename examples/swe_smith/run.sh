#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
case "$ROLE" in
  train|server|controller) [ "$#" -le 2 ] ;;
  trainer) [ "$#" -eq 1 ] ;;
  *) false ;;
esac || {
  echo "Usage: $0 train [experiment-name] | controller [machine]" >&2
  exit 2
}

requested_name="${2:-}"
cd "$(dirname "$0")/../.."
EXAMPLE_DIR="examples/swe_smith"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "$PWD/.venv/bin/python" ]; then
  # shellcheck disable=SC1091
  source "$PWD/.venv/bin/activate"
fi
has_gpu_allocation="${CUDA_VISIBLE_DEVICES+x}"
allocated_devices="${CUDA_VISIBLE_DEVICES:-}"
requested_config="${AGL_TRAIN_CONFIG:-}"
machine="machine"
if [ "$ROLE" = controller ] && [ -n "$requested_name" ]; then
  case "$requested_name" in
    *[!a-z0-9-]*) echo "ERROR: invalid machine name." >&2; exit 2 ;;
  esac
  machine="$requested_name"
fi
machine_file="${ARBAS_MACHINE_FILE:-$HOME/.config/arbas/$machine.env}"
if [ -f "$machine_file" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$machine_file"
  set +a
elif [ "$machine" != machine ] || [ -n "${ARBAS_MACHINE_FILE:-}" ]; then
  echo "ERROR: machine settings not found at $machine_file." >&2
  exit 1
fi
if [ -n "$has_gpu_allocation" ]; then
  export CUDA_VISIBLE_DEVICES="$allocated_devices"
fi

umask 077
AGL_SERVER_PORT="${AGL_SERVER_PORT:-8080}"
AGL_KEY="${AGL_KEY:-dummy}"
AGL_NAMESPACE="${AGL_NAMESPACE:-default}"
AGL_TRAIN_CONFIG="${requested_config:-${AGL_TRAIN_CONFIG:-$EXAMPLE_DIR/training.yaml}}"
output_root="${AGL_OUTPUT_DIR:-$PWD/outputs/swe-smith}"
LOCAL_SERVER_URL="http://127.0.0.1:$AGL_SERVER_PORT"
PUBLIC_HOST="${AGL_SERVER_PUBLIC_HOST:-127.0.0.1}"
SERVER_URL="http://${PUBLIC_HOST}:${AGL_SERVER_PORT}"
export AGL_KEY
mkdir -p "$output_root"

healthy() {
  curl -fsS --max-time 2 "$1/healthz" >/dev/null 2>&1
}

if [ "$ROLE" = train ] || [ "$ROLE" = server ]; then
  if healthy "$LOCAL_SERVER_URL"; then
    echo "ERROR: a server is already running on port $AGL_SERVER_PORT." >&2
    exit 1
  fi
  run_name="$(python - "$AGL_TRAIN_CONFIG" "$output_root" "$requested_name" <<'PY'
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from zoneinfo import ZoneInfo

from omegaconf import OmegaConf

from examples.swe_smith.pull_images import experiment_images
from examples.swe_smith.training_config import load_training_config

source = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])
config = load_training_config(source)
name = sys.argv[3] or config["experiment"]["name"]
if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", name):
    raise SystemExit("ERROR: use up to 48 lowercase letters, digits and hyphens for the experiment name.")
latest = root / "latest"
if latest.exists() and not latest.is_symlink():
    raise SystemExit(f"ERROR: {latest} already exists and is not a symlink.")
config["experiment"]["name"] = name
for field in ("train_files", "val_files"):
    path = Path(config.data[field])
    if not path.is_file():
        raise SystemExit(f"ERROR: {field} does not exist: {path}")
images = experiment_images(config)
stamp = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y%m%d-%H%M%S")
run_dir = Path(mkdtemp(prefix=f"{name}-{stamp}-", dir=root))
OmegaConf.save(config, run_dir / "training.yaml")
manifest = run_dir / ".active.json"
manifest.write_text(json.dumps({"run_id": run_dir.name, "images": images}), encoding="utf-8")
manifest.replace(root / "active.json")
link = run_dir / ".latest"
link.symlink_to(run_dir.name)
link.replace(latest)
print(run_dir.name)
PY
)"
  export AGL_RUN_DIR="$output_root/$run_name" AGL_RUN_NAME="$run_name"
  config_path="$AGL_RUN_DIR/training.yaml"
  proxy_settings="$(python -c '
import sys
from omegaconf import OmegaConf
config = OmegaConf.load(sys.argv[1])
actor = config.actor_rollout_ref
rollout = actor["rollout"]
include_experts = str(config.experiment.include_routed_experts).lower()
print(actor.model.path, rollout.temperature, rollout.val_kwargs.temperature, include_experts, sep="\t")
' "$config_path")"
  IFS=$'\t' read -r model train_temperature val_temperature include_experts <<< "$proxy_settings"
  server_command=(agl-server
    "port=$AGL_SERVER_PORT"
    "host=${AGL_SERVER_BIND:-127.0.0.1}"
    "key=\${oc.env:AGL_KEY}"
    "default_proxy.model_name=$model"
    "default_proxy.train.temperature=$train_temperature"
    "default_proxy.val.temperature=$val_temperature"
    "default_proxy.include_routed_experts=$include_experts"
    "hydra.run.dir=\${oc.env:AGL_RUN_DIR}" hydra.output_subdir=null 'hydra.job_logging.root.handlers=[console]')
  server_log="$AGL_RUN_DIR/$run_name-server.log"
  server_command=(python -u "$EXAMPLE_DIR/launch_trainer.py" --log-command "$server_log" "${server_command[@]}")
  echo "Experiment: $run_name"
  echo "Server log: $server_log"
  if [ "$ROLE" = server ]; then
    exec "${server_command[@]}"
  fi

  server_pid=""
  trainer_pid=""
  smoke_tmp=""
  stop_training() {
    if [ -n "$trainer_pid" ]; then
      kill -TERM "$trainer_pid" 2>/dev/null || true
      wait "$trainer_pid" 2>/dev/null || true
    fi
    if [ -n "$server_pid" ]; then
      kill -TERM "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    if [ -n "$smoke_tmp" ]; then rm -rf -- "$smoke_tmp"; fi
  }
  trap stop_training EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  "${server_command[@]}" &
  server_pid=$!
  for ((attempt = 0; attempt < 30; attempt++)); do
    if healthy "$LOCAL_SERVER_URL"; then break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if ! kill -0 "$server_pid" 2>/dev/null || ! healthy "$LOCAL_SERVER_URL"; then
    echo "ERROR: server did not start. See $server_log." >&2
    exit 1
  fi
  echo "On the Mac: bash examples/swe_smith/run.sh controller [machine]"
elif [ "$ROLE" = trainer ]; then
  if ! healthy "$LOCAL_SERVER_URL"; then
    echo "ERROR: start the server first, or use '$0 train'." >&2
    exit 1
  fi
  run_name="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$output_root/active.json")"
  export AGL_RUN_DIR="$output_root/$run_name" AGL_RUN_NAME="$run_name"
  config_path="$AGL_RUN_DIR/training.yaml"
fi

if [ "$ROLE" = train ] || [ "$ROLE" = trainer ]; then
  trainer_log="$AGL_RUN_DIR/$run_name-trainer.log"
  export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$AGL_RUN_DIR/tensorboard}"
  smoke="$(python -c 'import sys; from omegaconf import OmegaConf; print(str(OmegaConf.load(sys.argv[1]).experiment.check_smoke).lower())' "$config_path")"
  smoke_tmp=""
  smoke_args=()
  if [ "$smoke" = true ]; then
    smoke_args=(--smoke)
    scratch_parent="${AGL_SMOKE_TMPDIR:-/tmp}"
    if [ ! -d "$scratch_parent" ] || ! df -Pk "$scratch_parent" | awk 'NR == 2 { ok = $4 >= 16 * 1024 * 1024 && ($5 + 0) < 95 } END { exit !ok }'; then
      echo "ERROR: smoke needs at least 16 GiB free scratch space; set AGL_SMOKE_TMPDIR in machine.env." >&2
      exit 1
    fi
    smoke_tmp="$(mktemp -d "$scratch_parent/agl-smoke.XXXXXXXX")"
    if [ "$ROLE" = trainer ]; then trap 'rm -rf -- "$smoke_tmp"' EXIT; fi
    export TMPDIR="$smoke_tmp" RAY_TMPDIR="$smoke_tmp"
    mkdir -p "$TENSORBOARD_DIR"
    TENSORBOARD_DIR="$(mktemp -d "$TENSORBOARD_DIR/smoke.XXXXXXXX")"
  fi
  unset AGL_MODEL_NAME AGL_TRAIN_DATASET_PATH AGL_VAL_DATASET_PATH
  echo "Trainer log: $trainer_log"
  trainer_command=(python -u "$EXAMPLE_DIR/launch_trainer.py" "$trainer_log" "$smoke_tmp"
    "$EXAMPLE_DIR/train_smith_agent.py" --agl-base-url "$LOCAL_SERVER_URL"
    --config "$config_path" ${smoke_args[@]+"${smoke_args[@]}"})
  if [ "$ROLE" = trainer ]; then exec "${trainer_command[@]}"; fi
  "${trainer_command[@]}" &
  trainer_pid=$!
  while kill -0 "$trainer_pid" 2>/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "ERROR: server stopped during training. See $server_log." >&2
      exit 1
    fi
    sleep 1
  done
  wait "$trainer_pid"
  exit
fi

tunnel_dir=""
secret_file=""
active=""
cleanup() {
  if [ -n "$secret_file" ]; then rm -f "$secret_file"; fi
  if [ -n "$tunnel_dir" ]; then
    ssh -S "$tunnel_dir/control" -O exit "$AGL_SSH_TARGET" >/dev/null 2>&1 || true
    rm -rf "$tunnel_dir"
  fi
}
trap cleanup EXIT
if [ -n "${AGL_SSH_TARGET:-}" ] && ! healthy "$SERVER_URL"; then
  tunnel_dir="$(mktemp -d "${TMPDIR:-/tmp}/arbas-ssh.XXXXXXXX")"
  echo "Starting SSH tunnel to $AGL_SSH_TARGET"
  ssh -M -S "$tunnel_dir/control" -fnNT -o ExitOnForwardFailure=yes \
    -L "${AGL_SSH_BIND:-127.0.0.1}:$AGL_SERVER_PORT:127.0.0.1:${AGL_SSH_REMOTE_PORT:-8080}" "$AGL_SSH_TARGET"
fi
if ! healthy "$SERVER_URL"; then
  echo "ERROR: server is not reachable at $SERVER_URL." >&2
  exit 1
fi
if [ -z "${AGL_SERVER_AGENT_URL:-}" ] && [[ "$PUBLIC_HOST" = localhost || "$PUBLIC_HOST" = 127.* ]]; then
  echo "ERROR: set AGL_SERVER_AGENT_URL in $machine_file to an address reachable from the pods." >&2
  exit 1
fi
kubectl create namespace "$AGL_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-agent-scripts \
  --from-file="$EXAMPLE_DIR/hierarchical" --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-oracle-scripts \
  --from-file="$EXAMPLE_DIR/hierarchical/oracle" --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  secret_file="$(mktemp "${TMPDIR:-/tmp}/arbas-secret.XXXXXXXX")"
  printf '%s' "$OPENROUTER_API_KEY" > "$secret_file"
  kubectl -n "$AGL_NAMESPACE" create secret generic swe-smith-openrouter \
    --from-file="api-key=$secret_file" --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
  rm -f "$secret_file"
  secret_file=""
fi
if [ -n "${AGL_REMOTE_OUTPUT_DIR:-}" ]; then
  if [ -z "${AGL_SSH_TARGET:-}" ] || [[ "$AGL_REMOTE_OUTPUT_DIR" != /* ]]; then
    echo "ERROR: automatic image preparation needs AGL_SSH_TARGET and an absolute AGL_REMOTE_OUTPUT_DIR." >&2
    exit 1
  fi
  remote_file="$(python -c 'import shlex,sys; print(shlex.quote(sys.argv[1]))' "$AGL_REMOTE_OUTPUT_DIR/active.json")"
  if [ -n "$tunnel_dir" ]; then
    active="$(ssh -S "$tunnel_dir/control" "$AGL_SSH_TARGET" "cat $remote_file")"
  else
    # The path is quoted for the remote shell above.
    # shellcheck disable=SC2029
    active="$(ssh "$AGL_SSH_TARGET" "cat $remote_file")"
  fi
  echo "Preparing images for the current experiment"
  printf '%s' "$active" | python -c '
import json
import os
import re
import subprocess
import sys
active = json.load(sys.stdin)
images = active["images"]
command = [sys.executable, sys.argv[1]]
for image in images:
    command.extend(["--image", image])
env = os.environ.copy()
env.pop("OPENROUTER_API_KEY", None)
env.pop("AGL_KEY", None)
run_id = active["run_id"]
if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_id):
    raise SystemExit("ERROR: invalid run_id in active.json")
subprocess.run(command, check=True, env=env)
' "$EXAMPLE_DIR/pull_images.py"
fi
if [ -n "$active" ]; then
  run_name="$(printf '%s' "$active" | python -c 'import json,sys; print(json.load(sys.stdin)["run_id"])')"
  export AGL_RUN_DIR="$output_root/$run_name"
  mkdir -p "$AGL_RUN_DIR"
else
  AGL_RUN_DIR="$(mktemp -d "$output_root/controller-$(TZ=Europe/Moscow date +%Y%m%d-%H%M%S)-XXXXXX")"
  export AGL_RUN_DIR
  run_name="$(basename "$AGL_RUN_DIR")"
fi
controller_log="$AGL_RUN_DIR/$run_name-controller.log"
echo "Controller log: $controller_log"
python -u "$EXAMPLE_DIR/launch_trainer.py" --log-command "$controller_log" agl-controller \
  runner_type=k8s \
  agl_server.url="$SERVER_URL" \
  agl_server.agent_url="${AGL_SERVER_AGENT_URL:-$SERVER_URL}" \
  "agl_server.key=\${oc.env:AGL_KEY}" \
  hydra.job_logging.root.level=WARNING \
  k8s_runner.namespace="$AGL_NAMESPACE" \
  k8s_runner.ttl_after_finished="${AGL_JOB_TTL:-3600}" \
  "hydra.run.dir=\${oc.env:AGL_RUN_DIR}" hydra.output_subdir=null 'hydra.job_logging.root.handlers=[console]'
