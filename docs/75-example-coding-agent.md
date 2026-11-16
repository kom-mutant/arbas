# Coding Agent

| GPU | Model | Controller Mode | Trainer Mode | Code |
|---|---|---|---|---|
| 4× B200 | `Qwen/Qwen3.5-9B` | K8s | Sync and async | [Source](https://github.com/microsoft/agent-lightning/tree/main/examples/swe_smith) |

The Coding Agent example trains a software-engineering agent on SWE-smith tasks with `verl` and Agent Lightning >=v1.0. Each rollout runs as a Kubernetes Job inside a repository-specific image, edits an isolated checkout, executes tests, and reports the resulting reward to the AGL Gateway.

This example uses two machines:

- **Machine A — Kubernetes Controller machine:** connects to the Kubernetes cluster, prepares repository images in the node-accessible Docker runtime, and runs `agl-controller` to create rollout Jobs.
- **Machine B — GPU training machine:** provides the GPUs and runs both `agl-server` (the AGL Gateway) and the `verl` trainer with its model backend.

Machine B's AGL Gateway address must be reachable from Machine A and from the rollout pods in the Kubernetes cluster.

## Environment Preparation

On **Machine A (Kubernetes Controller machine)**, activate the project environment and install the dependency used to prepare repository images:

```bash
source .venv/bin/activate
uv pip install docker
```

Machine A also requires Docker, `kubectl`, and access to the Kubernetes cluster.

On **Machine B (GPU training machine)**, use the [training setup and launch instructions](../examples/swe_smith/README.md).

## Data Preparation

The provided splits are derived from the original SWE-smith dataset, which contains 59,136 executable software-engineering tasks from 128 Python repositories. We build the training data with the following filtering pipeline:

1. Remove tasks with an empty problem statement. The original release contains 18,033 such records.
2. Remove tasks whose corresponding problem branch is missing from the provided repository image. This affects 1,265 records.
3. Remove tasks requiring more than 200 tests, which avoids examples with prohibitively expensive test suites.
4. Run Qwen3.5-9B four times on every remaining candidate as a difficulty probe.
5. Remove tasks solved in all four probe rollouts because they provide little learning signal.
6. Retain tasks with a mixture of successful and failed probe rollouts, yielding approximately 5,000 examples.
7. Add a sample of 1,000 tasks that fail all four probes so the training set is not biased toward easier tasks.

The resulting data contains approximately 6,000 training examples and 400 validation examples. `train_dataset_mixed.jsonl` contains the mixed-difficulty training set, while `val_dataset_filtered.jsonl` contains the filtered validation set.

For experiments using the larger splits, download the pre-split dataset archive from
[Google Drive](https://drive.google.com/file/d/1q19DP53l4rldvBR2dkUhbaPI_mHVBVL1/view?usp=drive_link)
on the GPU machine. The current `training.yaml` points to these larger splits.

The larger splits contain these files:

- `examples/swe_smith/train_dataset_mixed.jsonl`
- `examples/swe_smith/val_dataset_filtered.jsonl`

When using `run.sh`, set custom paths in the experiment YAML under `data.train_files` and `data.val_files`.

## Repository Image Preparation

The controller can prepare the images automatically over SSH when `AGL_REMOTE_OUTPUT_DIR`
is configured, as described in the example README. Alternatively, on Machine A prepare
the images in the Docker daemon used by the Kubernetes nodes:

```bash
python examples/swe_smith/pull_images.py \
    --dataset examples/swe_smith/train_dataset_mixed.jsonl \
    --dataset examples/swe_smith/val_dataset_filtered.jsonl
```

This command installs the OpenAI SDK, Pydantic, and NumPy into each required SWE-smith base image and creates the `:arbas` tags expected by `job-template-openai.yaml`. Run it again if the datasets introduce new repository images.

## Training

Start the three roles in order: server, controller, trainer. All experiment settings
come from the GPU machine's YAML file. See the [SWE-smith instructions](../examples/swe_smith/README.md)
for the current commands, machine-specific connection settings, and log locations.

## Preventing Reward Hacking

A coding agent may obtain the reference fix without solving the task, for example by inspecting Git history, downloading upstream source code with `curl` or `wget`, installing the original package with `pip`, or using Python networking libraries such as `urllib`.

The SWE agent limits these reward-hacking paths in two ways:

- **Repository isolation:** before the agent starts, the harness checks out the task branch and moves `.git` outside the visible testbed. Agent commands that invoke Git, access the hidden Git metadata, install packages, download files, or modify the test harness are blocked.
- **Network isolation:** we strongly recommend adding a Kubernetes network policy that denies all outbound traffic from agent pods except connections to the AGL Gateway. Without this restriction, an agent may retrieve upstream source code or other external information and obtain reward without solving the task as intended.

The final reward is computed by running the task-specific `FAIL_TO_PASS` and `PASS_TO_PASS` tests inside the isolated repository environment. These controls are part of the training setup: weakening them can allow the agent to recover reference code and corrupt the reward signal.
