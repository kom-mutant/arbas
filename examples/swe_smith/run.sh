#!/usr/bin/env bash

set -euo pipefail
ROLE="${1:-}"
if [ "$ROLE" != "server" ] && [ "$ROLE" != "controller" ] && [ "$ROLE" != "trainer" ]; then
  echo "Usage: $0 {server|controller|trainer} [--smoke for controller and trainer]"
  echo "  server      → Machine B: agl-server (store + API only, no model backend)"
  echo "  controller  → Machine A: agent ConfigMap + agl-controller (k8s runner)"
  echo "  trainer     → Machine B: VERL trainer (enqueues rollouts)"
  echo ""
  echo "Start order: server → controller → trainer."
  exit 1
fi
shift || true
cd "$(dirname "$0")/../.."
EXAMPLE_DIR="examples/swe_smith"
AGL_SERVER_PORT="${AGL_SERVER_PORT:-8080}"
AGL_KEY="${AGL_KEY:-dummy}"
AGL_TRAIN_CONFIG="${AGL_TRAIN_CONFIG:-$EXAMPLE_DIR/training.yaml}"
export AGL_TRAIN_CONFIG
DEFAULT_TRAIN_SCRIPT="$EXAMPLE_DIR/train_smith_agent.py"
AGL_TRAIN_SCRIPT="${AGL_TRAIN_SCRIPT:-$DEFAULT_TRAIN_SCRIPT}"
AGL_INCLUDE_ROUTED_EXPERTS="${AGL_INCLUDE_ROUTED_EXPERTS:-false}"
AGL_NAMESPACE="${AGL_NAMESPACE:-default}"
PUBLIC_HOST="${AGL_SERVER_PUBLIC_HOST:-0.0.0.0}"
SERVER_URL="http://${PUBLIC_HOST}:${AGL_SERVER_PORT}"
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"

umask 077
run_dir="${AGL_RUN_DIR:-$PWD/outputs/swe-smith}"
mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
export AGL_KEY TENSORBOARD_DIR="${TENSORBOARD_DIR:-$run_dir/tensorboard}"
hydra_logs=("hydra.run.dir=$run_dir" hydra.output_subdir=null 'hydra.job_logging.root.handlers=[console]')
echo "Log: $run_dir/$ROLE.log"

run_logged() {
  "$@" 2>&1 | python -u -c '
import os
import sys

key = os.environ["AGL_KEY"]
last_progress = None
for line in sys.stdin:
    if "AglRolloutManager: completed=" in line:
        if line == last_progress:
            continue
        last_progress = line
    sys.stdout.write(line.replace(key, "[redacted]"))
' | tee -a "$run_dir/$ROLE.log"
}

if [ "$ROLE" = "server" ]; then
  if [ -z "${AGL_MODEL_NAME:-}" ]; then
    AGL_MODEL_NAME="$(python -c '
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).actor_rollout_ref.model.path)
' "$AGL_TRAIN_CONFIG")"
  fi
  echo "=== SWE-smith :: server (Machine B) ==="
  echo "  Public server URL: $SERVER_URL"
  echo "  No backend model is started here — this is just the store + REST API."
  echo "  Readiness criterion: GET ${SERVER_URL%/}/healthz returns 200."
  echo "=== Starting Agent Lightning server (no model backend) ==="
  run_logged agl-server \
    port="$AGL_SERVER_PORT" \
    host="${AGL_SERVER_BIND:-0.0.0.0}" \
    "key=\${oc.env:AGL_KEY}" \
    default_proxy.model_name="$AGL_MODEL_NAME" \
    default_proxy.include_routed_experts="$AGL_INCLUDE_ROUTED_EXPERTS" \
    "${hydra_logs[@]}"
elif [ "$ROLE" = "trainer" ]; then
  echo "=== SWE-smith :: trainer (Machine B) ==="
  echo "  Server: http://localhost:$AGL_SERVER_PORT"
  if ! curl -fsS --max-time 5 "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null 2>&1; then
    echo "ERROR: server not reachable at http://localhost:$AGL_SERVER_PORT — start './run.sh server' first."
    exit 1
  fi
  echo "=== Running SWE-smith training ==="
  if [ "$AGL_TRAIN_SCRIPT" != "$DEFAULT_TRAIN_SCRIPT" ]; then
    set -- --agl-key "$AGL_KEY" \
      --train-dataset-path "${AGL_TRAIN_DATASET_PATH:-$EXAMPLE_DIR/train_dataset_mixed.jsonl}" \
      --val-dataset-path "${AGL_VAL_DATASET_PATH:-$EXAMPLE_DIR/val_dataset_filtered.jsonl}" \
      --model "${AGL_MODEL_NAME:-Qwen/Qwen3.5-9B}" \
      --run-name "${AGL_RUN_NAME:-distributed}" "$@"
  fi
  run_logged python -u "$AGL_TRAIN_SCRIPT" \
    --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
    "$@"
elif [ "$ROLE" = "controller" ]; then
  settings="$(python -c '
import json
import os
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
agents = config.agents
if sys.argv[2:] == ["--smoke"]:
    agents = OmegaConf.merge(agents, config.smoke.agents)
for key, value in OmegaConf.to_container(agents, resolve=True).items():
    default = value if isinstance(value, str) else json.dumps(value)
    print(f"{key}={os.environ.get(key, default)}")
' "$AGL_TRAIN_CONFIG" "$@")"
  echo "=== SWE-smith :: controller (Machine A) ==="
  echo "  Connecting to server: $SERVER_URL"
  echo "  Namespace: $AGL_NAMESPACE"
  if ! curl -fsS --max-time 5 "${SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "WARNING: server not reachable at $SERVER_URL — start './run.sh server' on Machine B first."
  fi
  echo "=== Ensuring namespace '$AGL_NAMESPACE' exists ==="
  kubectl create namespace "$AGL_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
  echo "=== Creating agent scripts ConfigMap ==="
  kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-agent-scripts \
    --from-file="$EXAMPLE_DIR/hierarchical" \
    --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
  kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-oracle-scripts \
    --from-file="$EXAMPLE_DIR/hierarchical/oracle" \
    --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
  printf '%s\n' "$settings" | kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-settings \
    --from-env-file=/dev/stdin --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
  echo "=== Starting Agent Lightning controller (runner_type=k8s) ==="
  echo "  Next: start the trainer on Machine B with './run.sh trainer'."
  run_logged agl-controller \
    runner_type=k8s \
    agl_server.url="$SERVER_URL" \
    agl_server.agent_url="${AGL_SERVER_AGENT_URL:-$SERVER_URL}" \
    "agl_server.key=\${oc.env:AGL_KEY}" \
    hydra.job_logging.root.level=WARNING \
    k8s_runner.namespace="$AGL_NAMESPACE" \
    k8s_runner.ttl_after_finished=600 \
    "${hydra_logs[@]}"
fi
