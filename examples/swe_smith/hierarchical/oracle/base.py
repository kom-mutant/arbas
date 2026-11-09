"""Base selector interface and pickle checkpoints."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from .types import RoutingContext


class ModelSelector(ABC):
    """Base class for routing policies.

    Args:
        models: Names of the candidate models, in a fixed order.  By convention
            the list is ordered from strongest (most expensive) to weakest.
    """

    name: str = "selector"

    def __init__(self, models: Sequence[str]) -> None:
        self.models = list(models)
        self.index = {m: i for i, m in enumerate(self.models)}
        self.n_select = 0
        self.n_update = 0

    # --- the contract -------------------------------------------------------
    @abstractmethod
    def select(self, ctx: RoutingContext) -> str:
        """Return the model for a new program."""

    @abstractmethod
    def update(self, ctx: RoutingContext, model: str, reward: float) -> None:
        """Delayed feedback for a completed program routed to ``model``."""

    def state(self) -> Dict[str, Any]:
        """Serialisable view for the dashboard and for checkpoints."""
        return {"name": self.name, "models": self.models, "n_select": self.n_select, "n_update": self.n_update}

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ModelSelector":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
