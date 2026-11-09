#!/usr/bin/env bash
set -euo pipefail

# Run one GRPO update with several attempts on a real SWE-smith task.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

GPUS="${1:-1}"
MODEL="${2:-${AGL_MODEL_NAME:-Qwen/Qwen3.5-9B}}"
AGL_PORT="${AGL_SERVER_PORT:-8080}"
AGL_KEY="${AGL_KEY:-dummy}"
TRAIN_DATASET="${AGL_TRAIN_DATASET_PATH:-$SCRIPT_DIR/smoke_train.jsonl}"
VAL_DATASET="${AGL_VAL_DATASET_PATH:-$SCRIPT_DIR/smoke_val.jsonl}"
ROLLOUTS="${SMOKE_ROLLOUTS:-4}"
MAX_LEN="${MAX_LEN:-16384}"
HALF_LEN=$((MAX_LEN / 2))
GPU_MEM="${GPU_MEM:-0.7}"

echo "=========================================================="
echo " Starting 1-step smoke test training"
echo " GPUs:        $GPUS"
echo " Model:       $MODEL"
echo " MaxCtxLen:   $MAX_LEN (prompt=$HALF_LEN, resp=$HALF_LEN)"
echo " vLLM GPU%:   $GPU_MEM"
echo " Dataset:     $TRAIN_DATASET"
echo " Attempts:    $ROLLOUTS"
echo " Server URL:  http://localhost:$AGL_PORT"
echo " Logger:      console"
echo "=========================================================="

if ! curl -sf "http://localhost:$AGL_PORT/healthz" >/dev/null 2>&1; then
  echo "ERROR: agl-server не запущен на http://localhost:$AGL_PORT." >&2
  echo "Сначала в отдельном терминале/tmux запустите: ./examples/swe_smith/run.sh server" >&2
  exit 1
fi

python examples/swe_smith/train_smith_agent.py \
  --train-dataset-path "$TRAIN_DATASET" \
  --val-dataset-path "$VAL_DATASET" \
  --model "$MODEL" \
  --agl-base-url "http://localhost:$AGL_PORT" \
  --agl-key "$AGL_KEY" \
  --run-name smoke \
  trainer.n_gpus_per_node="$GPUS" \
  trainer.val_before_train=False \
  trainer.total_training_steps=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  'trainer.logger=[console]' \
  data.train_batch_size=1 \
  data.max_prompt_length="$HALF_LEN" \
  data.max_response_length="$HALF_LEN" \
  actor_rollout_ref.rollout.n="$ROLLOUTS" \
  actor_rollout_ref.rollout.max_model_len="$MAX_LEN" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM" \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  agentlightning.rollout_timeout_seconds=900 \
  agentlightning.trace_aggregator.trajectory_max_prompt_length="$HALF_LEN" \
  agentlightning.trace_aggregator.trajectory_max_response_length="$HALF_LEN" \
  "${@:3}"
