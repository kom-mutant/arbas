from __future__ import annotations

import hashlib
import re
from typing import Any, Dict

import numpy as np

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|\d+|[^\sA-Za-z0-9_]")


class HashingFeatures:
    """Hashed bag of words plus a few length features, L2-normalised, with a bias term.

    Zero dependencies, deterministic, good enough for the bandit to learn which
    task families each model handles well.  Swap in :class:`EmbeddingFeatures`
    for a real sentence embedding.
    """

    dim = 64
    max_tokens = 512

    def __call__(self, prompt: str, metadata: Dict[str, Any]) -> np.ndarray:
        x = np.zeros(self.dim, dtype=float)
        toks = _TOKEN.findall(prompt.lower())[: self.max_tokens]
        for t in toks:
            h = int(hashlib.blake2b(t.encode(), digest_size=4).hexdigest(), 16)
            x[h % (self.dim - 2)] += 1.0
        x /= max(np.linalg.norm(x), 1.0)
        x[-2] = min(len(prompt) / 4000.0, 1.0)  # length
        x[-1] = 1.0  # bias
        return x
