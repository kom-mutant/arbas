import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import APIError, OpenAI
from pydantic import ValidationError

from . import smith
from .agents import Orchestrator
from .config import Config, load_config
from .prompts import JUDGE_PROMPT
from .routing import ModelRouter
from .types import Judgment, SemanticScore, TaskResult

log = logging.getLogger(__name__)


def judge_connection(config: Config) -> tuple[str, str]:
    url = config.judge_base_url or config.subagent_base_url
    return url, config.judge_api_key or config.subagent_api_key


def evaluate_assignment(client: OpenAI, problem: str, result: TaskResult, config: Config) -> Judgment:
    settings = config.judge_settings()
    judgment = Judgment(model=settings.model)
    assignment = result.assignment
    evidence = {
        "original_issue": problem,
        "assignment": assignment.prompt,
        "context": assignment.context,
        "worker_instructions": "\n".join(m["content"] for m in result.messages if m["role"] == "system"),
        "report": result.summary,
        "command_observations": "\n\n".join(m["content"] for m in result.messages if m["role"] == "tool"),
        "patch_before": result.patch_before or "",
        "patch_after": result.patch_after or "",
        "status": result.status,
        "error": result.error,
    }
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            **settings.request_kwargs(),
        )
        choice = response.choices[0]
        judgment.response = choice.message.content or ""
        if choice.finish_reason != "stop" or choice.message.tool_calls:
            judgment.error = f"Incomplete judge response: {choice.finish_reason}"
        else:
            verdict = SemanticScore.model_validate_json(judgment.response)
            judgment.score, judgment.reason = verdict.score, verdict.reason
    except (APIError, ValidationError) as exc:
        judgment.error = str(exc)
    finally:
        judgment.latency_s = time.perf_counter() - t0

    if judgment.error:
        log.warning("Judge could not score %s: %s (took %.2fs)", result.agent_name, judgment.error, judgment.latency_s)
    else:
        log.info(
            "Judge %s evaluated %s in %.2fs (score=%s)",
            settings.model,
            result.agent_name,
            judgment.latency_s,
            judgment.score,
        )
    return judgment


def judge_assignments(
    problem: str, assignments: list[TaskResult], config: Config, address: tuple[str, str]
) -> list[float | None]:
    if not assignments:
        return []
    judge_url, judge_key = address
    scores = []
    with (
        OpenAI(base_url=judge_url, api_key=judge_key, timeout=config.api_timeout, max_retries=3) as client,
        ThreadPoolExecutor(max_workers=min(4, len(assignments))) as executor,
    ):
        futures = [executor.submit(evaluate_assignment, client, problem, result, config) for result in assignments]
        for result, future in zip(assignments, futures):
            judgment = future.result()
            result.judgment = judgment
            scores.append(judgment.score)
    return scores


def training_reward(raw_reward: float, resolved: bool, result: TaskResult, config: Config) -> float:
    if "/mode/train/" not in config.orchestrator_base_url:
        return raw_reward

    reward = smith.length_penalized_reward(
        raw_reward,
        result.steps_taken,
        config.orchestrator_steps,
        t0=int(os.environ.get("SMITH_LEN_PEN_T0", "80")),
        lam=float(os.environ.get("SMITH_LEN_PEN_LAMBDA", "0.1")),
        is_train=True,
    )
    hard_cap = int(os.environ.get("SMITH_PROMPT_PEN_HARD_CAP", "64000"))
    max_prompt_tokens = result.usage.max_prompt_tokens
    if max_prompt_tokens >= hard_cap:
        log.warning("max_prompt_tokens=%d >= hard_cap=%d (context near budget)", max_prompt_tokens, hard_cap)
    return smith.prompt_length_penalty(
        reward,
        max_prompt_tokens,
        soft_start=int(os.environ.get("SMITH_PROMPT_PEN_SOFT_START", "50000")),
        hard_cap=hard_cap,
        max_pen=float(os.environ.get("SMITH_PROMPT_PEN_MAX", "0.1")),
        is_train=True,
        solved=resolved,
    )


def main() -> int:
    started = time.perf_counter()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )

    problem = os.environ.get("AGL_TASK_INPUT", "").strip()
    if not problem:
        raise SystemExit("AGL_TASK_INPUT (problem statement) not set")
    eval_meta = json.loads(os.environ.get("AGL_EVAL_META", "") or "{}")
    if not eval_meta:
        eval_meta = smith.fetch_eval_meta()
    instance_id = eval_meta.get("instance_id", "")
    if not instance_id:
        raise SystemExit("eval meta unavailable: no instance_id (server fetch failed); aborting before rollout")

    config = load_config()
    judge_address = judge_connection(config) if config.oracle_models and config.oracle_feedback == "semantic" else None
    router = ModelRouter(config) if config.oracle_models else None
    max_turns = config.orchestrator_steps
    eval_timeout = int(os.environ.get("SMITH_EVAL_TIMEOUT", "600"))
    f2p_only = os.environ.get("SMITH_F2P_ONLY", "1").lower() not in ("0", "false", "no")

    log.info("SmithAgent start: instance=%s max_turns=%d", instance_id, max_turns)
    smith.checkout_bug_commit(instance_id)
    smith.relocate_git()
    with (
        OpenAI(
            base_url=config.orchestrator_base_url, api_key=config.orchestrator_api_key, timeout=config.api_timeout, max_retries=3
        ) as client,
        OpenAI(
            base_url=config.subagent_base_url,
            api_key=config.subagent_api_key,
            timeout=config.api_timeout,
            max_retries=3,
        ) as subagent_client,
    ):
        swarm = Orchestrator(config, client, subagent_client, router=router).run(problem)

    submitted = swarm.orchestrator.status == "completed"
    n_turns = swarm.orchestrator.steps_taken
    max_prompt_tokens = swarm.orchestrator.usage.max_prompt_tokens

    patch = smith.capture_patch()
    scores = []
    if judge_address is not None and swarm.assignments:
        judge_started = time.perf_counter()
        scores = judge_assignments(problem, swarm.assignments, config, judge_address)
        log.info(
            "Judge stage: assignments=%d scored=%d wall_s=%.2f",
            len(scores),
            sum(score is not None for score in scores),
            time.perf_counter() - judge_started,
        )
    raw_reward, resolved, reason, timed_out = smith.evaluate(eval_meta, eval_timeout, f2p_only=f2p_only)

    reward = training_reward(raw_reward, resolved, swarm.orchestrator, config)

    log.info(
        "done: mode=%s submitted=%s patch=%dB reward=%.3f raw_reward=%.3f n_turns=%d max_prompt_tokens=%d reason=%s",
        "train" if "/mode/train/" in config.orchestrator_base_url else "val",
        submitted,
        len(patch),
        reward,
        raw_reward,
        n_turns,
        max_prompt_tokens,
        reason,
    )

    if router is not None:
        if config.oracle_feedback == "episode":
            scores = [raw_reward] * len(swarm.assignments)
        router.feedback(scores, swarm.assignments)

    event_base = {"instance_id": instance_id, "repo": eval_meta.get("repo", "")}
    smith.post_event("agent_output", {**event_base, "patch": patch, "patch_size": len(patch), "submitted": submitted})
    smith.post_event(
        "reward",
        {
            **event_base,
            "value": reward,
            "raw_value": raw_reward,
            "resolved": resolved,
            "reason": reason,
            "n_turns": n_turns,
            "max_prompt_tokens": max_prompt_tokens,
            "eval_timeout": timed_out,
            "source": "agent",
        },
        retry=True,
    )
    output_dir = Path(os.environ.get("HIER_OUTPUT_DIR", "/tmp/hierarchical-smith"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (instance_id.replace("/", "__") + ".json")
    output = {
        "instance_id": instance_id,
        "patch": patch,
        "reward": reward,
        "raw_reward": raw_reward,
        "resolved": resolved,
        "reason": reason,
        "config": config.model_dump(exclude={"orchestrator_api_key", "subagent_api_key", "judge_api_key"}),
        **swarm.model_dump(mode="json"),
    }
    if router is not None:
        output["oracle"] = router.report()
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved episode: %s (rollout_wall_s=%.2f)", output_path, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
