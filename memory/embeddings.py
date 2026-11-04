"""Local embedding model wrapper using sentence-transformers with fallback."""

import hashlib
from typing import Any, Dict, List
from core.logger import logger

_model_instance = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    try:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading local embedding model ({model_name})...")
        _model_instance = SentenceTransformer(model_name)
        return _model_instance
    except Exception as e:
        # Loud on purpose: hash vectors carry NO meaning, so similarity, entity resolution,
        # abstraction, edge weights and the graph layout are all noise until this is fixed.
        logger.error(f"Could not load SentenceTransformer '{model_name}' ({e}). Falling back to a "
                     "deterministic HASH embedder: vectors are random, semantic memory is DEGRADED "
                     "(cognition.semantic_ok = false). Install sentence-transformers / the model to fix.")

        class FallbackEmbedder:
            semantic = False   # read by is_semantic(); surfaced as cognition.semantic_ok

            def encode(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = []
                for text in texts:
                    # Produce deterministic 384-dim vector from sha256
                    vec = []
                    for i in range(384):
                        h = hashlib.sha256(f"{text}_{i}".encode("utf-8")).hexdigest()
                        val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
                        vec.append(val)
                    embeddings.append(vec)
                return embeddings

        _model_instance = FallbackEmbedder()
        return _model_instance


def is_semantic(embed_fn) -> bool:
    """False when an embedding function is (or wraps) the random hash fallback."""
    if getattr(embed_fn, "semantic", True) is False:
        return False
    return getattr(getattr(embed_fn, "model", None), "semantic", True) is not False


class LocalEmbeddingFunction:
    """Full ChromaDB EmbeddingFunction-compatible wrapper (chromadb >= 1.5).

    Properly subclasses chromadb.EmbeddingFunction to implement:
      - __call__: embed a list of documents
      - embed_query: embed a query (delegates to __call__)
      - name(): stable string identifier
      - get_config() / build_from_config() / validate_config(): serialisation stubs
      - is_legacy(): returns False since all required methods are implemented
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self.model = get_embedding_model(model_name)

    @staticmethod
    def name() -> str:
        return "local-sentence-transformer"

    def __call__(self, input: List[str]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return list(embeddings)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """ChromaDB calls this for query-time embedding; delegates to __call__."""
        return self.__call__(input)

    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "LocalEmbeddingFunction":
        return LocalEmbeddingFunction(model_name=config.get("model_name", "all-MiniLM-L6-v2"))

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        pass

    def is_legacy(self) -> bool:
        """Returns False: all required EmbeddingFunction methods are implemented."""
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]
