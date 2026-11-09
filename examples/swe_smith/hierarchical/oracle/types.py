from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RoutingContext:
    """What a model selector sees when it picks a model for a new program."""

    program_id: str
    prompt: str = ""
    """Text of the program's initial request (first user message)."""
    features: Optional[np.ndarray] = None
    """Feature vector for contextual learners. Filled by a ``FeatureExtractor``."""
    task_type: Optional[str] = None
    """Task type inferred by the verifier selector (e.g. ``"swe"``)."""
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Anything else the caller wants to pass along (user id, budget, ...)."""
