"""FakeEmbedder — deterministic, fast embedder for tests (A3)."""
from __future__ import annotations
import hashlib
from typing import List


class FakeEmbeddingFunction:
    """Deterministic embedding function: hash-based, always 384-dim (matches all-MiniLM-L6-v2)."""

    def __call__(self, input: List[str]) -> List[List[float]]:
        result = []
        for text in input:
            # Produce a stable 384-dimensional vector from the text hash
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = []
            for i in range(384):
                seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
                vec.append((seed / 0xFFFFFFFF) * 2 - 1)  # [-1, 1]
            # Normalize
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            result.append([x / norm for x in vec])
        return result


FakeEmbedder = FakeEmbeddingFunction
