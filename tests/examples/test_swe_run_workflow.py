import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from omegaconf import OmegaConf

from agentlightning.controller.k8s_reconciler import build_job_spec
from agentlightning.schemas import Rollout, RolloutConfig, RolloutK8sConfig, RolloutLifecycleStatus
from examples.swe_smith.hierarchical.config import load_config
from examples.swe_smith.train_smith_agent import load_agent_settings, load_split_file
from examples.swe_smith.training_config import load_training_config

CONFIG = Path(__file__).resolve().parents[2] / "examples/swe_smith/training.yaml"


def test_settings_reach_agent_and_real_controller_template(tmp_path):
    source = OmegaConf.to_container(load_training_config(CONFIG))
    source["agents"]["SMITH_MAX_TURNS"] = 12
    source["agents"]["SUBAGENT_MAX_STEPS"] = 24
    source["agents"]["AGL_MAX_TOKENS"] = 3072
    source["agents"]["ORACLE_MODELS"] = ["test/model-a", "test/model-b"]
    source["agents"]["ORACLE_CHECKPOINT"] = "/custom/oracle.pkl"
    source["agents"]["SUBAGENT_MODEL"] = "literal {{ example }}"
    source["actor_rollout_ref"]["rollout"]["temperature"] = 0.55
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(source))
    settings = load_agent_settings(config_path, "test-run")
    with patch.dict(os.environ, {**settings, "AGL_OPENAI_BASE_URL": "https://gateway.invalid/v1",
                                 "OPENROUTER_API_KEY": "router-secret"}, clear=True):
        config = load_config()
    assert config.subagent_steps == 24
    assert config.orchestrator_steps == 12
    assert config.subagent_model.model == "literal {{ example }}"
    assert config.orchestrator_model.max_tokens == 3072
    assert config.orchestrator_model.temperature == 0.55
    assert config.oracle_models == ["test/model-a", "test/model-b"]
    assert config.f2p_only is True
    assert config.judge_model.reasoning is None
    template = (CONFIG.parent / "job-template-openai.yaml").read_text()
    dataset = load_split_file(str(CONFIG.parent / "smoke_train.jsonl"), agent_settings=settings)
    rollout = Rollout(
        rollout_id="test-id", input=dataset[0],
        config=RolloutConfig(k8s=RolloutK8sConfig(job_template=template)),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0),
    )
    controller = OmegaConf.create({"agl_server": {"url": "http://server:8080", "key": "agl-secret"},
                                   "k8s_runner": {"namespace": "default", "ttl_after_finished": 60}})
    job = build_job_spec(rollout, controller)
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert f'$AGL_RUN_NAME-{job["metadata"]["name"]}.log' in container["args"][0]
    volumes = {item["name"]: item for item in job["spec"]["template"]["spec"]["volumes"]}
    assert volumes["agent-scripts"]["configMap"]["name"] == "swe-smith-agent-scripts"
    assert volumes["oracle-scripts"]["configMap"]["name"] == "swe-smith-oracle-scripts"
    assert volumes["rollout-logs"]["hostPath"]["path"] == "/agl-logs"
    assert {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}["rollout-logs"] == "/agl-logs"
    env = {item["name"]: item for item in container["env"]}
    assert env["SUBAGENT_MODEL"]["value"] == "literal {{ example }}"
    assert env["SUBAGENT_MAX_STEPS"]["value"] == "24"
    assert env["ORACLE_CHECKPOINT"]["value"] == "/custom/oracle.pkl"
    assert env["AGL_KEY"]["value"] == "agl-secret"
    assert env["OPENROUTER_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "swe-smith-openrouter"


def test_dataset_settings_survive_arrow_serialization(tmp_path):
    pa = pytest.importorskip("pyarrow")
    source = OmegaConf.to_container(load_training_config(CONFIG))
    source["agents"]["ORACLE_MODELS"] = ["test/model-a", "test/model-b"]
    source["agents"].pop("ORACLE_CHECKPOINT", None)
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(source))
    settings = load_agent_settings(config_path, "test-run")
    assert settings["ORACLE_CHECKPOINT"] == "/agl-logs/test-run/oracle/linucb.pkl"
    rows = load_split_file(str(CONFIG.parent / "smoke_train.jsonl"), agent_settings=settings)
    restored = pa.Table.from_pylist(rows).to_pylist()
    assert restored[0]["arbas_agent_settings"] == settings


def executable(path, contents):
    path.write_text("#!/bin/sh\nset -eu\n" + contents + "\n")
    path.chmod(0o755)


@pytest.mark.parametrize("smoke", [True, False])
def test_train_launches_both_processes_with_named_logs(tmp_path, smoke):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = shlex.quote(sys.executable)
    executable(
        bin_dir / "python",
        f'case "$*" in *launch_trainer.py*--log-command*) exec {interpreter} "$@";;\n'
        '*launch_trainer.py*)\n'
        '  test -e "$MOCK_SERVER_READY"\n'
        '  printf "%s\\n" "$@" > "$MOCK_TRAINER_ARGS"\n'
        '  printf "%s\\n" "$CUDA_VISIBLE_DEVICES" > "$3"\n'
        '  exit "$MOCK_TRAINER_STATUS";;\n'
        f'*) exec {interpreter} "$@";; esac',
    )
    executable(bin_dir / "curl", '[ -e "$MOCK_SERVER_READY" ]')
    executable(
        bin_dir / "agl-server",
        'trap \'rm -f "$MOCK_SERVER_READY"; exit 0\' TERM\n'
        'printf "%s\\n" "$@" > "$MOCK_SERVER_ARGS"\n'
        'echo "server started with $AGL_KEY"\n'
        'touch "$MOCK_SERVER_READY"\n'
        'while :; do sleep 0.1; done',
    )
    executable(bin_dir / "df", "printf 'Filesystem blocks used available capacity mounted\\n/dev/test 999999999 0 999999999 1%% /tmp\\n'")
    machine = tmp_path / "machine.env"
    config = yaml.safe_load(CONFIG.read_text())
    config["base"] = str(CONFIG.parent / config["base"])
    config["experiment"]["check_smoke"] = smoke
    config["experiment"]["include_routed_experts"] = not smoke
    config["actor_rollout_ref"]["rollout"]["temperature"] = 0.85
    config["actor_rollout_ref"]["rollout"]["val_kwargs"] = {"temperature": 0.25}
    for split in ("train", "val"):
        dataset = CONFIG.parent / f"smoke_{split}.jsonl"
        (tmp_path / dataset.name).write_bytes(dataset.read_bytes())
        config["data"][f"{split}_files"] = dataset.name
    config["data"]["max_train_instances"] = 1
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config))
    machine_config = config_path if smoke else tmp_path / "other.yaml"
    machine.write_text(f'AGL_KEY=shared-test-key\nAGL_SERVER_PORT=8080\nAGL_TRAIN_CONFIG="{machine_config}"\n'
                       f'CUDA_VISIBLE_DEVICES=0,1\nAGL_SMOKE_TMPDIR="{tmp_path}"\n')
    env = {**os.environ, "PATH": f'{bin_dir}{os.pathsep}{os.environ["PATH"]}', "VIRTUAL_ENV": sys.prefix,
           "ARBAS_MACHINE_FILE": str(machine), "AGL_OUTPUT_DIR": str(tmp_path / "runs"),
           "MOCK_SERVER_READY": str(tmp_path / "ready"), "MOCK_SERVER_ARGS": str(tmp_path / "server-args"),
           "MOCK_TRAINER_ARGS": str(tmp_path / "trainer-args"), "CUDA_VISIBLE_DEVICES": "5",
           "MOCK_TRAINER_STATUS": "0" if smoke else "7"}
    env.pop("AGL_TRAIN_CONFIG", None)
    if not smoke:
        env["AGL_TRAIN_CONFIG"] = str(config_path)
    cmd = ["bash", "examples/swe_smith/run.sh"]
    (tmp_path / "ready").touch()
    duplicate = subprocess.run([*cmd, "train"], cwd=root, env=env, capture_output=True, text=True, timeout=15)
    assert duplicate.returncode != 0 and "already running" in duplicate.stderr
    (tmp_path / "ready").unlink()
    result = subprocess.run([*cmd, "train", "h200-2b"], cwd=root, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == int(env["MOCK_TRAINER_STATUS"]), result.stdout + result.stderr
    server_args = (tmp_path / "server-args").read_text()
    assert "default_proxy.train.temperature=0.85" in server_args
    assert "default_proxy.val.temperature=0.25" in server_args
    assert f"default_proxy.include_routed_experts={str(not smoke).lower()}" in server_args
    active = json.loads((tmp_path / "runs/active.json").read_text())["run_id"]
    assert active.startswith("h200-2b-")
    run_dir = tmp_path / "runs" / active
    assert (tmp_path / "runs/latest").resolve() == run_dir
    snapshot = run_dir / "training.yaml"
    saved_config = yaml.safe_load(snapshot.read_text())
    assert saved_config["experiment"]["name"] == "h200-2b"
    assert saved_config["data"]["train_files"] == str(tmp_path / "smoke_train.jsonl")
    assert "base" not in saved_config
    assert saved_config["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["MAX_JOBS"] == "2"
    assert saved_config["actor_rollout_ref"]["rollout"]["engine_kwargs"]["vllm"] == {
        "enable_auto_tool_choice": True,
        "moe_backend": "triton",
        "gdn_prefill_backend": "triton",
    }
    assert (run_dir / f"{active}-server.log").read_text().strip() == "server started with [redacted]"
    assert (run_dir / f"{active}-trainer.log").read_text().strip() == "5"
    trainer_args = (tmp_path / "trainer-args").read_text()
    assert str(snapshot) in trainer_args
    assert ("--smoke" in trainer_args) is smoke
    assert "run.sh controller" in result.stdout
    assert not (tmp_path / "ready").exists()
    assert not list(tmp_path.glob("agl-smoke.*"))



def test_controller_loads_secret_and_manages_optional_tunnel(tmp_path):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    executable(bin_dir / "curl", '[ -e "$MOCK_TUNNEL_READY" ]')
    executable(bin_dir / "ssh", 'printf "%s\\n" "$*" >> "$MOCK_SSH_LOG"\n'
               'case "$*" in *"-O exit"*) rm -f "$MOCK_TUNNEL_READY";; *) touch "$MOCK_TUNNEL_READY";; esac')
    executable(bin_dir / "kubectl", 'printf "%s\\n" "$*" >> "$MOCK_KUBECTL_LOG"\n'
               'case "$*" in *"apply -f -"*) cat >/dev/null;; *"--dry-run=client"*) echo "kind: Secret";; esac')
    executable(bin_dir / "agl-controller", 'printf "%s\\n" "$@" > "$MOCK_CONTROLLER_ARGS"')
    machine = tmp_path / "machine.env"
    machine.write_text("AGL_SERVER_PORT=18080\nAGL_SERVER_AGENT_URL=http://host.docker.internal:18080\n"
                       "AGL_SSH_TARGET=gpu-host\nAGL_SSH_BIND=0.0.0.0\nAGL_KEY=shared-test-key\n"
                       "OPENROUTER_API_KEY=router-test-secret\nAGL_TRAIN_CONFIG=/does/not/exist\n")
    env = {**os.environ, "PATH": f'{bin_dir}{os.pathsep}{os.environ["PATH"]}', "VIRTUAL_ENV": sys.prefix,
           "ARBAS_MACHINE_FILE": str(machine), "AGL_OUTPUT_DIR": str(tmp_path / "runs"),
           "MOCK_TUNNEL_READY": str(tmp_path / "tunnel"), "MOCK_SSH_LOG": str(tmp_path / "ssh-log"),
           "MOCK_KUBECTL_LOG": str(tmp_path / "kubectl-log"), "MOCK_CONTROLLER_ARGS": str(tmp_path / "controller-args")}
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(["bash", "examples/swe_smith/run.sh", "controller"], cwd=root, env=env,
                            check=True, capture_output=True, text=True)
    assert "router-test-secret" not in result.stdout + result.stderr
    kubectl_log = (tmp_path / "kubectl-log").read_text()
    assert "router-test-secret" not in kubectl_log
    secret_path = re.search(r"--from-file=api-key=(\S+)", kubectl_log)
    assert secret_path and not Path(secret_path.group(1)).exists()
    assert "-O exit" in (tmp_path / "ssh-log").read_text()
    assert not (tmp_path / "tunnel").exists()
    assert "runner_type=k8s" in (tmp_path / "controller-args").read_text()
    machine.write_text(machine.read_text().replace("OPENROUTER_API_KEY=router-test-secret\n", ""))
    (tmp_path / "kubectl-log").write_text("")
    subprocess.run(["bash", "examples/swe_smith/run.sh", "controller"], cwd=root, env=env,
                   check=True, capture_output=True, text=True)
    assert "create secret generic" not in (tmp_path / "kubectl-log").read_text()


@pytest.mark.parametrize("existing_tunnel", [True, False])
def test_controller_prepares_server_images_without_local_dataset(tmp_path, existing_tunnel):
    root = CONFIG.parents[2]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    executable(bin_dir / "curl", '[ -e "$MOCK_TUNNEL_READY" ]')
    executable(bin_dir / "ssh", 'case "$*" in *"-O exit"*) rm -f "$MOCK_TUNNEL_READY";; '
               '*"cat "*) printf \'{"run_id":"x","images":["repo/first","repo/second"]}\';; '
               '*) touch "$MOCK_TUNNEL_READY";; esac')
    executable(bin_dir / "kubectl", 'case "$*" in *"apply -f -"*) cat >/dev/null;; '
               '*"--dry-run=client"*) echo "kind: Secret";; esac')
    executable(bin_dir / "agl-controller", 'echo ready')
    (tmp_path / "docker.py").write_text(
        "import os\n"
        "class Images:\n"
        "    def get(self, name):\n"
        "        assert 'OPENROUTER_API_KEY' not in os.environ and 'AGL_KEY' not in os.environ\n"
        "        with open(os.environ['MOCK_IMAGES_LOG'], 'a') as f: f.write(name + '\\n')\n"
        "        return object()\n"
        "class Client:\n"
        "    images = Images()\n"
        "def from_env(): return Client()\n"
    )
    machine = tmp_path / "machine.env"
    machine.write_text("AGL_SERVER_PORT=18080\nAGL_SERVER_AGENT_URL=http://host.docker.internal:18080\n"
                       "AGL_SSH_TARGET=gpu-host\nAGL_SSH_BIND=0.0.0.0\nAGL_KEY=shared-test-key\n"
                       "OPENROUTER_API_KEY=router-test-secret\nAGL_REMOTE_OUTPUT_DIR='/path with spaces/outputs'\n")
    env = {**os.environ, "PATH": f'{bin_dir}{os.pathsep}{os.environ["PATH"]}', "VIRTUAL_ENV": sys.prefix,
           "PYTHONPATH": f'{tmp_path}{os.pathsep}{os.environ.get("PYTHONPATH", "")}',
           "ARBAS_MACHINE_FILE": str(machine), "AGL_OUTPUT_DIR": str(tmp_path / "runs"),
           "MOCK_TUNNEL_READY": str(tmp_path / "tunnel"), "MOCK_IMAGES_LOG": str(tmp_path / "images")}
    if existing_tunnel:
        (tmp_path / "tunnel").touch()
    result = subprocess.run(["bash", "examples/swe_smith/run.sh", "controller"], cwd=root, env=env,
                            check=True, capture_output=True, text=True)
    assert "Prepared 2 SWE-smith :arbas images" in result.stdout
    assert (tmp_path / "runs/x/x-controller.log").read_text().strip() == "ready"
    assert set((tmp_path / "images").read_text().splitlines()) == {"repo/first:arbas", "repo/second:arbas"}
    assert "router-test-secret" not in result.stdout + result.stderr
    assert (tmp_path / "tunnel").exists() is existing_tunnel


def test_qwen35_chat_template_is_prefix_preserving():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=False, lstrip_blocks=False)
    env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
    env.filters["items"] = lambda d: list(d.items())
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    template_path = Path(__file__).resolve().parents[2] / "examples/swe_smith/qwen35_chat_template.jinja"
    tmpl = env.from_string(template_path.read_text(encoding="utf-8"))
    tools = [{"type": "function", "function": {"name": "create_subagent", "parameters": {"type": "object"}}}]

    msgs = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Issue"},
        {
            "role": "assistant",
            "reasoning_content": "First step\n\n",
            "content": None,
            "tool_calls": [{"function": {"name": "create_subagent", "arguments": {"name": "research"}}}],
        },
        {"role": "tool", "content": '{"agent_name": "research_1"}'},
        {
            "role": "assistant",
            "reasoning": None,
            "content": None,
            "tool_calls": [{"function": {"name": "create_subagent", "arguments": {"name": "fix"}}}],
        },
        {"role": "tool", "content": '{"agent_name": "fix_2"}'},
        {"role": "user", "content": "Retry one tool call."},
        {"role": "assistant", "reasoning_content": "Final\n", "content": "Done\n"},
    ]

    p0 = tmpl.render(messages=msgs[:2], tools=tools, add_generation_prompt=True)
    ctx0 = (
        p0
        + "First step\n\n</think>\n\n<tool_call>\n<function=create_subagent>\n"
        + "<parameter=name>\nresearch\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    )
    p1 = tmpl.render(messages=msgs[:4], tools=tools, add_generation_prompt=True)
    assert p1.startswith(ctx0)
    ctx1 = (
        p1
        + "</think>\n\n<tool_call>\n<function=create_subagent>\n"
        + "<parameter=name>\nfix\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
    )
    p2 = tmpl.render(messages=msgs[:7], tools=tools, add_generation_prompt=True)
    assert p2.startswith(ctx1)
    p3 = tmpl.render(messages=msgs, tools=tools, add_generation_prompt=False)
    assert p3.startswith(p2)
    assert "<think>\nFirst step\n\n</think>" in p3

    no_think_messages = json.loads(json.dumps(msgs[:4]))
    no_think_messages[2].pop("reasoning_content")
    no_think_p0 = tmpl.render(
        messages=no_think_messages[:2], tools=tools, add_generation_prompt=True, enable_thinking=False
    )
    no_think_p1 = tmpl.render(
        messages=no_think_messages, tools=tools, add_generation_prompt=True, enable_thinking=False
    )
    assert no_think_p0.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert no_think_p1.startswith(no_think_p0)
