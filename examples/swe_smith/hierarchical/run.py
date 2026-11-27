from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIError, OpenAI
from pydantic import ValidationError

from .agents import Orchestrator
from .config import Config, load_config
from .oracle import ModelRouter
from .prompts import JUDGE_PROMPT
from .sandbox import ContainerSandbox, Evaluation, select_eval_nodes
from .types import AgentRunResult, AssignmentResult, Judgment, SemanticScore, SwarmResult

log = logging.getLogger(__name__)


@dataclass
class EpisodeOutcome:
    run_id: str | None
    job_name: str
    instance_id: str
    repo: str
    started_at: str
    mode: str
    submitted: bool = False
    patch: str | None = None
    patch_error: str | None = None
    evaluation: Evaluation | None = None
    reward: float | None = None
    reward_published: bool = False
    turns: int = 0
    max_prompt_tokens: int = 0
    judged_assignments: int = 0
    timings_s: dict[str, float | None] = field(
        default_factory=lambda: {"setup": None, "solve": None, "judge": None, "evaluation": None, "rollout_wall": None}
    )

    def metrics(self) -> dict[str, Any]:
        ev = self.evaluation
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "job_name": self.job_name,
            "instance_id": self.instance_id,
            "reward": self.reward,
            "raw_reward": ev.reward if ev is not None else None,
            "reward_computed": self.reward is not None,
            "reward_published": self.reward_published,
            "resolved": ev.resolved if ev is not None else False,
            "submitted": self.submitted,
            "reason": ev.reason if ev is not None else "evaluation did not complete",
            "eval_timeout": ev.timed_out if ev is not None else False,
            "patch_error": self.patch_error,
            "turns": self.turns,
            "max_prompt_tokens": self.max_prompt_tokens,
            "judged_assignments": self.judged_assignments,
            "timings_s": self.timings_s,
        }


def post_event(event_type: str, data: dict[str, Any], retry: bool = False) -> bool:
    event_url = os.environ.get("AGL_EVENT_URL")
    if not event_url:
        log.warning("AGL_EVENT_URL not set; skip %s event", event_type)
        return False
    body = json.dumps({"event_type": event_type, "data": data}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if agl_key := os.environ.get("AGL_KEY") or os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {agl_key}"
    request = urllib.request.Request(event_url, data=body, headers=headers, method="POST")
    attempts = 4 if retry else 1
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read()
            return True
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            fatal_http = isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code != 429
            if fatal_http or attempt == attempts:
                log.warning("failed to post %s event: %s", event_type, exc)
                return False
            backoff = 2**attempt
            log.warning("failed to post %s event (attempt %d): %s; retrying in %ds", event_type, attempt, exc, backoff)
            time.sleep(backoff)
    return False


def judge_connection(config: Config) -> tuple[str, str]:
    url = config.judge.base_url or config.subagent.base_url
    return url, config.judge_api_key or config.subagent_api_key


def evaluate_assignment(client: OpenAI, problem: str, result: AssignmentResult, config: Config) -> Judgment:
    settings = config.judge
    judgment = Judgment(model=settings.model)
    if result.assignment is None:
        judgment.error = "Missing task assignment metadata"
        return judgment
    evidence = {
        "original_issue": problem,
        "assignment": result.assignment.prompt,
        "context": result.assignment.context,
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
            result.judgment = future.result()
            scores.append(result.judgment.score)
    return scores


def training_reward(evaluation: Evaluation, result: AgentRunResult, config: Config) -> float:
    """Compute scalar RL reward: optional partial F2P credit, minus turn and prompt-bloat penalties on solved tasks."""
    if config.mode != "train":
        return evaluation.reward
    cfg = config.reward
    reward = evaluation.partial_reward if cfg.partial_credit else evaluation.reward
    if not evaluation.resolved:
        return reward

    if result.steps_taken > cfg.len_pen_t0:
        turn_frac = min(
            (result.steps_taken - cfg.len_pen_t0) / max(1, config.orchestrator.max_turns - cfg.len_pen_t0),
            1.0,
        )
        reward = 1.0 - cfg.len_pen_lambda * turn_frac

    max_prompt = result.usage.max_prompt_tokens
    if max_prompt >= cfg.prompt_pen_hard_cap:
        log.warning("max_prompt_tokens=%d >= hard_cap=%d (context near budget)", max_prompt, cfg.prompt_pen_hard_cap)
    if max_prompt > cfg.prompt_pen_soft_start:
        prompt_frac = min(
            (max_prompt - cfg.prompt_pen_soft_start) / max(1, cfg.prompt_pen_hard_cap - cfg.prompt_pen_soft_start),
            1.0,
        )
        reward -= cfg.prompt_pen_max * prompt_frac
    return reward


def publish_outcome(outcome: EpisodeOutcome) -> bool:
    if outcome.evaluation is None or outcome.reward is None:
        return False
    base = {"instance_id": outcome.instance_id, "repo": outcome.repo}
    patch_text = outcome.patch or ""
    post_event(
        "agent_output",
        {
            **base,
            "patch": patch_text,
            "patch_size": len(patch_text),
            "patch_error": outcome.patch_error,
            "submitted": outcome.submitted,
        },
    )
    delivered = post_event(
        "reward",
        {
            **base,
            "value": outcome.reward,
            "raw_value": outcome.evaluation.reward,
            "resolved": outcome.evaluation.resolved,
            "reason": outcome.evaluation.reason,
            "n_turns": outcome.turns,
            "max_prompt_tokens": outcome.max_prompt_tokens,
            "eval_timeout": outcome.evaluation.timed_out,
            "source": "agent",
        },
        retry=True,
    )
    return bool(delivered)


def write_episode_file(
    output_dir: Path,
    outcome: EpisodeOutcome,
    config: Config,
    swarm: SwarmResult | None,
    router: ModelRouter | None,
) -> Path:
    output_path = output_dir / (outcome.instance_id.replace("/", "__") + ".json")
    metrics = outcome.metrics()
    metrics["assignment_count"] = len(swarm.assignments) if swarm is not None else 0
    payload: dict[str, Any] = {
        **metrics,
        "patch": outcome.patch,
        "config": config.model_dump(exclude={"orchestrator_api_key", "subagent_api_key", "judge_api_key"}),
    }
    if swarm is not None:
        payload.update(swarm.model_dump(mode="json"))
    if router is not None:
        payload["oracle"] = router.report()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("saved episode: %s (rollout_wall=%.2fs)", output_path, outcome.timings_s["rollout_wall"])
    log.info("episode_metrics=%s", json.dumps(metrics, ensure_ascii=False))
    return output_path


@contextmanager
def _episode_log(path: Path) -> Iterator[None]:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(formatter)
    file = logging.FileHandler(path, encoding="utf-8")
    file.setFormatter(formatter)
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(stdout)
    root.addHandler(file)
    try:
        yield
    finally:
        root.removeHandler(stdout)
        root.removeHandler(file)
        file.close()
        root.setLevel(previous_level)


def _rollout_job_name() -> str:
    rollout_id = os.environ.get("AGL_ROLLOUT_ID", "").strip()
    attempt_id = os.environ.get("AGL_ATTEMPT_ID", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", rollout_id) or not re.fullmatch(r"[A-Za-z0-9._-]+", attempt_id):
        raise ValueError("AGL_ROLLOUT_ID and AGL_ATTEMPT_ID are required and must be safe identifiers")
    return f"agl-rollout-{rollout_id}-{attempt_id}"


def main(sandbox: ContainerSandbox | None = None) -> None:
    """Run one rollout on the host; only shell commands and pytest execute inside the container."""
    run_dir = Path(os.environ["AGL_RUN_DIR"]).resolve()
    job_name = _rollout_job_name()
    output_dir = run_dir / "rollouts" / job_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with _episode_log(output_dir / f"{job_name}.log"):
        _run_episode(run_dir, job_name, output_dir, sandbox)


def _run_episode(run_dir: Path, job_name: str, output_dir: Path, sandbox: ContainerSandbox | None) -> None:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()  # noqa: UP017

    problem = os.environ.get("AGL_TASK_INPUT", "").strip()
    if not problem:
        raise SystemExit("AGL_TASK_INPUT (problem statement) not set")
    raw_eval_meta = os.environ.get("AGL_EVAL_META", "").strip()
    if not raw_eval_meta:
        raise SystemExit("AGL_EVAL_META not set")
    eval_meta = json.loads(raw_eval_meta)
    instance_id = eval_meta.get("instance_id", "")
    if not instance_id:
        raise SystemExit("AGL_EVAL_META missing instance_id")

    config = load_config()
    oracle_cfg = config.oracle
    if oracle_cfg.checkpoint == "auto":
        oracle_cfg.checkpoint = str(run_dir / "oracle" / "linucb.pkl")
    judge_address = judge_connection(config) if oracle_cfg.models and oracle_cfg.feedback == "semantic" else None
    router = ModelRouter(config) if oracle_cfg.models else None
    fail_to_pass, pass_to_pass = select_eval_nodes(eval_meta, f2p_only=config.sandbox.f2p_only)

    swarm: SwarmResult | None = None
    outcome = EpisodeOutcome(
        run_id=os.environ.get("AGL_RUN_NAME"),
        job_name=job_name,
        instance_id=instance_id,
        repo=eval_meta.get("repo", ""),
        started_at=started_at,
        mode=config.mode,
    )
    container: AbstractContextManager[ContainerSandbox] = (
        nullcontext(sandbox)
        if sandbox is not None
        else ContainerSandbox(
            os.environ["IMAGE_NAME"],
            job_name,
            run_dir=run_dir,
            cpus=config.sandbox.cpus,
            memory=config.sandbox.memory,
            ttl_seconds=max(config.sandbox.eval_timeout + 3600, 7200),
        )
    )

    try:
        log.info("SmithAgent start: instance=%s max_turns=%d", instance_id, config.orchestrator.max_turns)
        with container as box:
            box.setup(instance_id, fail_to_pass + pass_to_pass)
            solve_started = time.perf_counter()
            outcome.timings_s["setup"] = solve_started - started
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
                swarm = Orchestrator(config, client, subagent_client, box, router=router).run(problem)
            outcome.timings_s["solve"] = time.perf_counter() - solve_started
            outcome.submitted = swarm.orchestrator.status == "completed"
            outcome.turns = swarm.orchestrator.steps_taken
            outcome.max_prompt_tokens = swarm.orchestrator.usage.max_prompt_tokens

            try:
                outcome.patch = box.capture_patch()
            except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
                outcome.patch_error = str(exc)
                log.error("patch capture failed: %s", exc)

            eval_started = time.perf_counter()
            outcome.evaluation = box.evaluate(eval_meta, config.sandbox.eval_timeout, f2p_only=config.sandbox.f2p_only)
            outcome.timings_s["evaluation"] = time.perf_counter() - eval_started

        outcome.reward = training_reward(outcome.evaluation, swarm.orchestrator, config)

        scores: list[float | None] = []
        if judge_address is not None and swarm.assignments:
            judge_started = time.perf_counter()
            scores = judge_assignments(problem, swarm.assignments, config, judge_address)
            outcome.timings_s["judge"] = time.perf_counter() - judge_started
            outcome.judged_assignments = sum(score is not None for score in scores)
            log.info(
                "Judge stage: assignments=%d scored=%d wall_s=%.2f",
                len(scores),
                outcome.judged_assignments,
                outcome.timings_s["judge"],
            )
        if router is not None:
            if oracle_cfg.feedback == "episode":
                scores = [outcome.evaluation.reward] * len(swarm.assignments)
            router.feedback(scores, swarm.assignments)
    finally:
        outcome.timings_s["rollout_wall"] = time.perf_counter() - started
        outcome.reward_published = publish_outcome(outcome)
        log.info(
            "done: mode=%s submitted=%s patch=%s patch_error=%s reward=%s published=%s turns=%d reason=%s",
            outcome.mode,
            outcome.submitted,
            f"{len(outcome.patch)}B" if outcome.patch is not None else "error",
            bool(outcome.patch_error),
            f"{outcome.reward:.3f}" if outcome.reward is not None else "none",
            outcome.reward_published,
            outcome.turns,
            outcome.evaluation.reason if outcome.evaluation is not None else "incomplete",
        )
        write_episode_file(output_dir, outcome, config, swarm, router)


class RolloutWorker:
    """Entrypoint that ``LocalReconciler`` instantiates for every rollout."""

    def run(self) -> None:
        main()
