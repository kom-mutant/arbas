from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from .base import ModelSelector
from .types import RoutingContext


def _ctx_features(ctx: RoutingContext, d: int) -> np.ndarray:
    x = np.asarray(ctx.features, dtype=float)
    if x.shape != (d,):
        raise ValueError(f"feature shape {x.shape} != selector dim ({d},)")
    return x


class LinUCB(ModelSelector):
    """Disjoint LinUCB (Li et al. 2010): one ridge-regression arm per model.

    Args:
        models: candidate models.
        dim: feature dimension (match your ``FeatureExtractor.dim``).
        alpha_ucb: exploration width.  Called ``alpha_UCB`` in the paper's appendix
            to distinguish it from the reward's cost weight.
        lam: ridge regulariser.
        seed: tie-breaking RNG seed.
    """

    name = "linucb"

    def __init__(self, models: Sequence[str], dim: int = 64, alpha_ucb: float = 1.0, lam: float = 1.0, seed: int = 0) -> None:
        super().__init__(models)
        self.d = int(dim)
        self.alpha_ucb = float(alpha_ucb)
        self.rng = np.random.default_rng(seed)
        k = len(self.models)
        self.A = np.stack([lam * np.eye(self.d) for _ in range(k)])
        self.b = np.zeros((k, self.d))
        self._Ainv = np.stack([np.eye(self.d) / lam for _ in range(k)])
        self.pulls = np.zeros(k, dtype=int)

    def _theta(self, j: int) -> np.ndarray:
        return self._Ainv[j] @ self.b[j]

    def scores(self, ctx: RoutingContext) -> Dict[str, float]:
        x = _ctx_features(ctx, self.d)
        out = {}
        for j, m in enumerate(self.models):
            Ai = self._Ainv[j]
            out[m] = float(self._theta(j) @ x + self.alpha_ucb * math.sqrt(max(float(x @ Ai @ x), 0.0)))
        return out

    def select(self, ctx: RoutingContext) -> str:
        self.n_select += 1
        p = self.scores(ctx)
        vals = np.array([p[m] for m in self.models])
        best = np.flatnonzero(vals >= vals.max() - 1e-12)
        j = int(self.rng.choice(best))
        self.pulls[j] += 1
        return self.models[j]

    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        self.n_update += 1
        j = self.index[model]
        x = _ctx_features(ctx, self.d)
        self.A[j] += np.outer(x, x)
        self.b[j] += reward * x
        self._Ainv[j] = np.linalg.inv(self.A[j])

    def state(self) -> Dict[str, Any]:
        s = super().state()
        s.update({"alpha_ucb": self.alpha_ucb, "dim": self.d, "pulls": {m: int(self.pulls[j]) for j, m in enumerate(self.models)}})
        return s
