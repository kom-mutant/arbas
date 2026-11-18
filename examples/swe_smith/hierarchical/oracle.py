from __future__ import annotations

import fcntl
import hashlib
import math
import pickle
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .config import Config
from .types import AgentSpec, AssignmentResult, ModelSettings, TaskAssignment

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+|[^\sA-Za-z0-9_]")


class HashingFeatures:
    """Hashed bag of words plus length and bias features (64-dim, L2-normalised)."""

    dim = 64
    max_tokens = 512

    def __call__(self, prompt: str) -> np.ndarray:
        x = np.zeros(self.dim, dtype=float)
        for token in _TOKEN.findall(prompt.lower())[: self.max_tokens]:
            digest = int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16)
            x[digest % (self.dim - 2)] += 1.0
        x /= max(float(np.linalg.norm(x)), 1.0)
        x[-2] = min(len(prompt) / 4000.0, 1.0)
        x[-1] = 1.0
        return x


def oracle_reward(score: float, cost: float, cost_weight: float, cost_ref: float) -> float:
    """Compute scalar bandit reward: (1 - w) * score - w * min(cost / cost_ref, 1)."""
    if not 0.0 <= cost_weight <= 1.0:
        raise ValueError("cost_weight must be in [0, 1]")
    if cost_ref <= 0:
        raise ValueError("cost_ref must be positive")
    cost_n = min(max(cost, 0.0) / cost_ref, 1.0)
    return (1.0 - cost_weight) * score - cost_weight * cost_n


def _check_features(features: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.shape != (dim,):
        raise ValueError(f"feature shape {x.shape} != selector dim ({dim},)")
    return x


class LinUCB:
    """Disjoint LinUCB (Li et al. 2010): one ridge-regression arm per model."""

    name = "linucb"

    def __init__(
        self,
        models: Sequence[str],
        dim: int = 64,
        alpha_ucb: float = 1.0,
        lam: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("models must be non-empty")
        self.index = {model: i for i, model in enumerate(self.models)}
        self.d = int(dim)
        self.alpha_ucb = float(alpha_ucb)
        self.rng = np.random.default_rng(seed)
        k = len(self.models)
        self.A = np.stack([lam * np.eye(self.d) for _ in range(k)])
        self.b = np.zeros((k, self.d))
        self._Ainv = np.stack([np.eye(self.d) / lam for _ in range(k)])
        self.pulls = np.zeros(k, dtype=int)
        self.n_select = 0
        self.n_update = 0

    def scores(self, features: np.ndarray) -> dict[str, float]:
        x = _check_features(features, self.d)
        out: dict[str, float] = {}
        for j, model in enumerate(self.models):
            a_inv = self._Ainv[j]
            theta = a_inv @ self.b[j]
            out[model] = float(theta @ x + self.alpha_ucb * math.sqrt(max(float(x @ a_inv @ x), 0.0)))
        return out

    def select(self, features: np.ndarray) -> str:
        self.n_select += 1
        ucb = self.scores(features)
        vals = np.array([ucb[model] for model in self.models])
        best = np.flatnonzero(vals >= vals.max() - 1e-12)
        choice = int(self.rng.choice(best))
        self.pulls[choice] += 1
        return self.models[choice]

    def update(self, features: np.ndarray, model: str, reward: float) -> None:
        self.n_update += 1
        j = self.index[model]
        x = _check_features(features, self.d)
        self.A[j] += np.outer(x, x)
        self.b[j] += reward * x
        self._Ainv[j] = np.linalg.inv(self.A[j])

    def state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "models": self.models,
            "n_select": self.n_select,
            "n_update": self.n_update,
            "alpha_ucb": self.alpha_ucb,
            "dim": self.d,
            "pulls": {model: int(self.pulls[j]) for j, model in enumerate(self.models)},
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> LinUCB:
        with open(path, "rb") as f:
            return pickle.load(f)


class ModelRouter:
    """Routes subagent assignments across candidate models and persists LinUCB state."""

    def __init__(self, config: Config) -> None:
        oracle = config.oracle
        self.models = list(oracle.models)
        self.checkpoint = Path(oracle.checkpoint)
        self.learn = oracle.learn and config.mode != "val"
        self.seed = oracle.seed
        self.cost_weight = oracle.cost_weight
        self.cost_ref = oracle.cost_ref
        self.score_source = "semantic_judge" if oracle.feedback == "semantic" else "final_swe_raw_reward"
        self.features = HashingFeatures()
        self.selector = self._load()
        self.episode_id = uuid4().hex
        self.choices: list[tuple[np.ndarray, dict[str, Any]]] = []

    def _load(self) -> LinUCB:
        if not self.checkpoint.exists():
            return LinUCB(self.models, dim=self.features.dim, seed=self.seed)
        selector = LinUCB.load(self.checkpoint)
        if selector.models != self.models:
            raise ValueError("oracle.models differs from the checkpoint pool; use a new oracle.checkpoint")
        return selector

    @contextmanager
    def _training_state(self) -> Iterator[LinUCB]:
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.selector = self._load()
            yield self.selector
            temporary = self.checkpoint.with_suffix(".tmp")
            self.selector.save(temporary)
            temporary.replace(self.checkpoint)

    def select(
        self,
        problem: str,
        agent: AgentSpec,
        assignment: TaskAssignment,
        defaults: ModelSettings,
    ) -> ModelSettings:
        prompt = (
            f"Profile: {agent.profile}\nTask: {assignment.prompt}\n"
            f"Context: {assignment.context}\nOriginal issue: {problem}"
        )
        features = self.features(prompt)
        if self.learn:
            with self._training_state() as selector:
                model = selector.select(features)
        else:
            model = self.selector.select(features)
        decision: dict[str, Any] = {
            "program_id": f"{self.episode_id}:{len(self.choices)}",
            "agent_name": agent.name,
            "profile": agent.profile,
            "model": model,
            "prompt": prompt,
            "features": features.tolist(),
        }
        self.choices.append((features, decision))
        return defaults.model_copy(update={"model": model})

    def feedback(self, scores: Sequence[float | None], results: Sequence[AssignmentResult]) -> None:
        for (_, decision), result, score in zip(self.choices, results, scores, strict=True):
            cost = result.usage.completion_tokens
            reward = oracle_reward(score, cost, self.cost_weight, self.cost_ref) if score is not None else None
            decision.update(
                score=score,
                cost=cost,
                reward=reward,
                status=result.status,
                learned=False,
            )
        scored = [(features, decision) for features, decision in self.choices if decision["reward"] is not None]
        if not self.learn or not scored:
            return

        with self._training_state() as selector:
            for features, decision in scored:
                selector.update(features, decision["model"], decision["reward"])
                decision["learned"] = True

    def report(self) -> dict[str, Any]:
        return {
            "learning": self.learn,
            "checkpoint": str(self.checkpoint),
            "score_source": self.score_source,
            "cost_unit": "completion_tokens",
            "decisions": [decision for _, decision in self.choices],
            "selector": self.selector.state(),
        }
