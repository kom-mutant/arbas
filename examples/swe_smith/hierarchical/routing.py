import fcntl
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .config import Config
from .oracle.bandit import LinUCB
from .oracle.features import HashingFeatures
from .oracle.reward import Reward
from .oracle.types import RoutingContext
from .types import AgentSpec, ModelSettings, TaskAssignment, TaskResult


class ModelRouter:

    def __init__(self, config: Config) -> None:
        self.models = config.oracle_models
        self.checkpoint = Path(config.oracle_checkpoint)
        self.learn = config.oracle_learn and "/mode/val/" not in config.orchestrator_base_url
        self.seed = config.oracle_seed
        self.score_source = "semantic_judge" if config.oracle_feedback == "semantic" else "final_swe_raw_reward"
        self.features = HashingFeatures()
        self.reward = Reward(alpha=config.oracle_cost_weight, cost_ref=config.oracle_cost_ref)
        self.selector = self._load()
        self.episode_id = uuid4().hex
        self.choices: list[tuple[RoutingContext, dict]] = []

    def _load(self) -> LinUCB:
        if not self.checkpoint.exists():
            return LinUCB(self.models, dim=self.features.dim, seed=self.seed)
        selector = LinUCB.load(str(self.checkpoint))
        if selector.models != self.models:
            raise ValueError("ORACLE_MODELS differs from the checkpoint pool; use a new ORACLE_CHECKPOINT")
        return selector

    @contextmanager
    def _training_state(self):
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.selector = self._load()
            yield self.selector
            temporary = self.checkpoint.with_suffix(".tmp")
            self.selector.save(str(temporary))
            temporary.replace(self.checkpoint)

    def select(
        self, problem: str, agent: AgentSpec, assignment: TaskAssignment, defaults: ModelSettings
    ) -> ModelSettings:
        prompt = (
            f"Profile: {agent.profile}\nTask: {assignment.prompt}\n"
            f"Context: {assignment.context}\nOriginal issue: {problem}"
        )
        ctx = RoutingContext(
            program_id=f"{self.episode_id}:{len(self.choices)}",
            prompt=prompt,
            features=self.features(prompt, {}),
            task_type=agent.profile,
        )
        if self.learn:
            with self._training_state() as selector:
                model = selector.select(ctx)
        else:
            model = self.selector.select(ctx)
        decision = {
            "program_id": ctx.program_id,
            "agent_name": agent.name,
            "profile": agent.profile,
            "model": model,
            "prompt": prompt,
            "features": ctx.features.tolist(),
        }
        self.choices.append((ctx, decision))
        return defaults.model_copy(update={"model": model})

    def feedback(self, scores: list[float | None], results: list[TaskResult]) -> None:
        for (_, decision), result, score in zip(self.choices, results, scores, strict=True):
            cost = result.usage.completion_tokens
            decision.update(
                score=score,
                cost=cost,
                reward=self.reward(score, cost) if score is not None else None,
                status=result.status,
                learned=False,
            )
        scored = [(ctx, decision) for ctx, decision in self.choices if decision["reward"] is not None]
        if not self.learn or not scored:
            return

        with self._training_state() as selector:
            for ctx, decision in scored:
                selector.update(ctx, decision["model"], decision["reward"])
                decision["learned"] = True

    def report(self) -> dict:
        return {
            "learning": self.learn,
            "checkpoint": str(self.checkpoint),
            "score_source": self.score_source,
            "cost_unit": "completion_tokens",
            "decisions": [decision for _, decision in self.choices],
            "selector": self.selector.state(),
        }
