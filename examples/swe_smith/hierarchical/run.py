from __future__ import annotations

import http.client
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI
from pydantic import ValidationError

from . import smith
from .agents import Orchestrator
from .config import Config, load_config
from .oracle import ModelRouter
from .prompts import JUDGE_PROMPT
from .types import AgentRunResult, AssignmentResult, Judgment, SemanticScore, SwarmResult

log = logging.getLogger(__name__)


def post_event(event_type: str, data: dict[str, Any], retry: bool = False) -> None:
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log.warning("AGL_EVENT_URL not set; skip %s event", event_type)
        return
    body = json.dumps({"event_type": event_type, "data": data}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    agl_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY")
    if agl_key:
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(event_url, data=body, headers=headers, method="POST")
    attempts = 4 if retry else 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read()
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            fatal_http = isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code != 429
            if fatal_http or attempt == attempts:
                log.warning("failed to post %s event: %s", event_type, exc)
                return
            backoff = 2**attempt
            log.warning(
                "failed to post %s event (attempt %d): %s; retrying in %ds",
                event_type,
                attempt,
                exc,
                backoff,
            )
            time.sleep(backoff)


def fetch_eval_meta() -> dict[str, Any]:
    event_url = os.environ.get("AGL_EVENT_URL", "")
    match = re.match(r"^(?P<base>.*)/api/rollouts/(?P<rid>[^/]+)/attempt/", event_url)
    if not match:
        log.error("cannot derive rollout id from AGL_EVENT_URL=%r; eval meta unavailable", event_url)
        return {}
    url = f"{match.group('base')}/api/rollouts/{match.group('rid')}"
    headers: dict[str, str] = {}
    agl_key = os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY")
    if agl_key:
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    payload = None
    attempts = 4
    for i in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as exc:
            log.warning("fetch eval meta failed (attempt %d/%d): %s", i + 1, attempts, exc)
            if i < attempts - 1:
                time.sleep(min(2**i, 8))
    if payload is None:
        log.error("failed to fetch eval meta from server after %d attempts", attempts)
        return {}
    inp = (payload.get("rollout") or {}).get("input") or {}
    return {
        "instance_id": inp.get("instance_id", ""),
        "repo": inp.get("repo", ""),
        "FAIL_TO_PASS": inp.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": inp.get("PASS_TO_PASS", []),
    }


def length_penalized_reward(
    reward: float,
    n_turns: int,
    max_turns: int,
    *,
    t0: int,
    lam: float,
    is_train: bool,
) -> float:
    """Apply a linear turn penalty only to solved training rollouts."""
    if not is_train or reward < 1.0 or n_turns <= t0:
        return reward
    span = max(1, max_turns - t0)
    frac = min((n_turns - t0) / span, 1.0)
    return 1.0 - lam * frac


def prompt_length_penalty(
    reward: float,
    max_prompt_tokens: int,
    *,
    soft_start: int,
    hard_cap: int,
    max_pen: float,
    is_train: bool,
    solved: bool,
) -> float:
    """Apply a linear context-bloat penalty only to solved training rollouts."""
    if not is_train or not solved or max_prompt_tokens <= soft_start:
        return reward
    if max_prompt_tokens >= hard_cap:
        return reward - max_pen
    span = max(1, hard_cap - soft_start)
    frac = min((max_prompt_tokens - soft_start) / span, 1.0)
    return reward - max_pen * frac


def judge_connection(config: Config) -> tuple[str, str]:
    url = config.judge.base_url or config.subagent.base_url
    return url, config.judge_api_key or config.subagent_api_key


def evaluate_assignment(client: OpenAI, problem: str, result: AssignmentResult, config: Config) -> Judgment:
    settings = config.judge
    judgment = Judgment(model=settings.model)
    assignment = result.assignment
    if assignment is None:
        judgment.error = "Missing task assignment metadata"
        return judgment
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
        judgment.usage.add(response.usage)
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
    problem: str,
    assignments: list[AssignmentResult],
    config: Config,
    address: tuple[str, str],
) -> list[float | None]:
    if not assignments:
        return []
    judge_url, judge_key = address
    scores: list[float | None] = []
    with (
        OpenAI(base_url=judge_url, api_key=judge_key, timeout=config.sandbox.api_timeout, max_retries=3) as client,
        ThreadPoolExecutor(max_workers=min(4, len(assignments))) as executor,
    ):
        futures = [executor.submit(evaluate_assignment, client, problem, result, config) for result in assignments]
        for result, future in zip(assignments, futures, strict=True):
            judgment = future.result()
            result.judgment = judgment
            scores.append(judgment.score)
    return scores


def training_reward(evaluation: smith.Evaluation, result: AgentRunResult, config: Config) -> float:
    if config.mode != "train":
        return evaluation.reward

    reward_cfg = config.reward
    raw_reward = evaluation.partial_reward if reward_cfg.partial_credit else evaluation.reward
    reward = length_penalized_reward(
        raw_reward,
        result.steps_taken,
        config.orchestrator.max_turns,
        t0=reward_cfg.len_pen_t0,
        lam=reward_cfg.len_pen_lambda,
        is_train=True,
    )
    hard_cap = reward_cfg.prompt_pen_hard_cap
    max_prompt_tokens = result.usage.max_prompt_tokens
    if max_prompt_tokens >= hard_cap:
        log.warning("max_prompt_tokens=%d >= hard_cap=%d (context near budget)", max_prompt_tokens, hard_cap)
    return prompt_length_penalty(
        reward,
        max_prompt_tokens,
        soft_start=reward_cfg.prompt_pen_soft_start,
        hard_cap=hard_cap,
        max_pen=reward_cfg.prompt_pen_max,
        is_train=True,
        solved=evaluation.resolved,
    )


def publish_result(
    eval_meta: dict[str, Any],
    patch: str,
    submitted: bool,
    evaluation: smith.Evaluation,
    reward: float,
    n_turns: int,
    max_prompt_tokens: int,
) -> None:
    event_base = {"instance_id": eval_meta.get("instance_id", ""), "repo": eval_meta.get("repo", "")}
    post_event("agent_output", {**event_base, "patch": patch, "patch_size": len(patch), "submitted": submitted})
    post_event(
        "reward",
        {
            **event_base,
            "value": reward,
            "raw_value": evaluation.reward,
            "resolved": evaluation.resolved,
            "reason": evaluation.reason,
            "n_turns": n_turns,
            "max_prompt_tokens": max_prompt_tokens,
            "eval_timeout": evaluation.timed_out,
            "source": "agent",
        },
        retry=True,
    )


def save_result(
    *,
    instance_id: str,
    started_at: str,
    patch: str,
    submitted: bool,
    evaluation: smith.Evaluation,
    reward: float,
    n_turns: int,
    max_prompt_tokens: int,
    judged_assignments: int,
    timings: dict[str, float | None],
    config: Config,
    swarm: SwarmResult,
    router: ModelRouter | None,
) -> Path:
    output_dir = Path(os.environ.get("HIER_OUTPUT_DIR", "/tmp/hierarchical-smith"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (instance_id.replace("/", "__") + ".json")
    episode_metrics = {
        "run_id": os.environ.get("AGL_RUN_NAME"),
        "started_at": started_at,
        "job_name": os.environ.get("AGL_ROLLOUT_JOB_NAME"),
        "instance_id": instance_id,
        "reward": reward,
        "raw_reward": evaluation.reward,
        "resolved": evaluation.resolved,
        "submitted": submitted,
        "reason": evaluation.reason,
        "eval_timeout": evaluation.timed_out,
        "turns": n_turns,
        "max_prompt_tokens": max_prompt_tokens,
        "assignment_count": len(swarm.assignments),
        "judged_assignments": judged_assignments,
        "timings_s": timings,
    }
    output = {
        **episode_metrics,
        "patch": patch,
        "config": config.model_dump(exclude={"orchestrator_api_key", "subagent_api_key", "judge_api_key"}),
        **swarm.model_dump(mode="json"),
    }
    if router is not None:
        output["oracle"] = router.report()
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved episode: %s (rollout_wall_s=%.2f)", output_path, timings["rollout_wall"])
    log.info("episode_metrics=%s", json.dumps(episode_metrics, ensure_ascii=False))
    return output_path


def main() -> int:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    problem = os.environ.get("AGL_TASK_INPUT", "").strip()
    if not problem:
        raise SystemExit("AGL_TASK_INPUT (problem statement) not set")
    eval_meta = json.loads(os.environ.get("AGL_EVAL_META", "") or "{}")
    if not eval_meta:
        eval_meta = fetch_eval_meta()
    instance_id = eval_meta.get("instance_id", "")
    if not instance_id:
        raise SystemExit("eval meta unavailable: no instance_id (server fetch failed); aborting before rollout")

    config = load_config()
    oracle_cfg = config.oracle
    judge_address = judge_connection(config) if oracle_cfg.models and oracle_cfg.feedback == "semantic" else None
    router = ModelRouter(config) if oracle_cfg.models else None

    log.info("SmithAgent start: instance=%s max_turns=%d", instance_id, config.orchestrator.max_turns)
    smith.checkout_bug_commit(instance_id)
    smith.relocate_git()
    solve_started = time.perf_counter()
    with (
        OpenAI(
            base_url=config.orchestrator_base_url,
            api_key=config.orchestrator_api_key,
            timeout=config.sandbox.api_timeout,
            max_retries=3,
        ) as client,
        OpenAI(
            base_url=config.subagent.base_url,
            api_key=config.subagent_api_key,
            timeout=config.sandbox.api_timeout,
            max_retries=3,
        ) as subagent_client,
    ):
        swarm = Orchestrator(config, client, subagent_client, router=router).run(problem)
    solve_wall_s = time.perf_counter() - solve_started

    submitted = swarm.orchestrator.status == "completed"
    n_turns = swarm.orchestrator.steps_taken
    max_prompt_tokens = swarm.orchestrator.usage.max_prompt_tokens
    patch = smith.capture_patch()

    evaluation: smith.Evaluation | None = None
    reward = 0.0
    scores: list[float | None] = []
    judge_wall_s: float | None = None
    evaluation_wall_s: float | None = None
    judged_assignments = 0

    def _run_evaluation() -> tuple[smith.Evaluation, float, float]:
        eval_t0 = time.perf_counter()
        ev = smith.evaluate(eval_meta, config.sandbox.eval_timeout, f2p_only=config.sandbox.f2p_only)
        wall = time.perf_counter() - eval_t0
        rew = training_reward(ev, swarm.orchestrator, config)
        return ev, wall, rew

    try:
        if judge_address is not None and swarm.assignments:
            judge_started = time.perf_counter()
            scores = judge_assignments(problem, swarm.assignments, config, judge_address)
            judge_wall_s = time.perf_counter() - judge_started
            judged_assignments = sum(score is not None for score in scores)
            log.info(
                "Judge stage: assignments=%d scored=%d wall_s=%.2f",
                len(scores),
                judged_assignments,
                judge_wall_s,
            )

        evaluation, evaluation_wall_s, reward = _run_evaluation()

        if router is not None:
            if oracle_cfg.feedback == "episode":
                scores = [evaluation.reward] * len(swarm.assignments)
            router.feedback(scores, swarm.assignments)
    finally:
        try:
            if evaluation is None:
                evaluation, evaluation_wall_s, reward = _run_evaluation()
            log.info(
                "done: mode=%s submitted=%s patch=%dB reward=%.3f raw_reward=%.3f "
                "n_turns=%d max_prompt_tokens=%d reason=%s",
                config.mode,
                submitted,
                len(patch),
                reward,
                evaluation.reward,
                n_turns,
                max_prompt_tokens,
                evaluation.reason,
            )
            publish_result(eval_meta, patch, submitted, evaluation, reward, n_turns, max_prompt_tokens)
        finally:
            eval_result = evaluation or smith.Evaluation(0.0, False, "evaluation failed", False)
            timings: dict[str, float | None] = {
                "setup": solve_started - started,
                "solve": solve_wall_s,
                "judge": judge_wall_s,
                "evaluation": evaluation_wall_s,
                "rollout_wall": time.perf_counter() - started,
            }
            save_result(
                instance_id=instance_id,
                started_at=started_at,
                patch=patch,
                submitted=submitted,
                evaluation=eval_result,
                reward=reward,
                n_turns=n_turns,
                max_prompt_tokens=max_prompt_tokens,
                judged_assignments=judged_assignments,
                timings=timings,
                config=config,
                swarm=swarm,
                router=router,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
