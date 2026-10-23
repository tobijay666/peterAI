"""WordHashEmbedder — deterministic embedder where shared words mean similar vectors.

The hash-based FakeEmbedder gives every text an unrelated random vector, which is
fine for storage tests but useless for anything that depends on *similarity*
(entity resolution, abstraction, dedupe). This one is a bag-of-words: each
lower-cased token (and a crude stem) adds weight to a few hashed dimensions, so
"henna tattoo designs" and "henna designs" score high and "saree" vs "henna"
score near zero. No network, no model download, stable across runs.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List

_DIM = 384
_STOP = {"a", "an", "the", "is", "are", "of", "and", "or", "in", "on", "to", "for", "what",
         "how", "why", "who", "does", "do", "with", "about", "it", "its", "be", "was", "by"}


def _stem(w: str) -> str:
    for suf in ("ing", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    return w


def _slots(token: str):
    h = hashlib.md5(token.encode()).digest()
    return [(h[i] | (h[i + 1] << 8)) % _DIM for i in range(0, 6, 2)]


class WordHashEmbedder:
    def __call__(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        out = []
        for text in input:
            vec = [0.0] * _DIM
            toks = [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP]
            for t in toks:
                for slot in _slots(_stem(t)):
                    vec[slot] += 1.0
            if not toks:
                vec[0] = 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    # ChromaDB EmbeddingFunction protocol
    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "test-word-hash"

    def get_config(self) -> Dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "WordHashEmbedder":
        return WordHashEmbedder()

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        pass

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]
