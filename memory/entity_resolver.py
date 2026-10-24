"""EntityResolver — query -> graph entity nodes (spec Phase 1.2 node lookup).

1. Extract 1–3 candidate entities from the query: one cheap LLM call returning
   JSON, falling back to a stop-word heuristic when there is no router or the
   reply is unusable.
2. Resolve each term: alias table (exact) first, then the entity index
   (semantic) with a similarity floor.

Unresolved terms are knowledge gaps.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.cognition_config import cognition_config
from core.logger import logger
from memory.graph_store import GraphMemory
from memory.vector_store import VectorMemory

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has",
    "had", "will", "would", "could", "should", "may", "might", "can", "shall", "must", "how", "what",
    "when", "where", "why", "who", "whom", "which", "whose", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that", "these",
    "those", "and", "or", "but", "so", "for", "of", "to", "in", "on", "at", "by", "from", "with", "about",
    "into", "as", "than", "then", "please", "help", "tell", "explain", "give", "make", "made", "get", "let",
    "want", "need", "like", "just", "also", "very", "much", "more", "most", "some", "any", "all", "no",
    "not", "peter", "sir", "know", "mean", "means", "used", "use", "uses", "using", "work", "works",
    "thing", "things", "there", "here", "if", "can't", "cant", "describe", "show", "find", "look", "up",
    "anything", "something", "everything", "kind", "kinds", "type", "types", "many", "few", "best",
    "long", "last", "lasts", "often", "far", "old", "new", "good", "bad", "happen", "happens", "happened",
    "worn", "wear", "wears", "served", "serve", "cooked", "cook", "ties", "tie", "tied", "meaning",
    "go", "goes", "take", "takes", "come", "comes", "put", "see", "say", "says", "said", "way", "ways",
    "get", "gets", "made", "done", "called", "call", "happening", "today", "really", "exactly",
    "love", "loves", "loved", "loving", "hate", "hates", "hated", "enjoy", "enjoys", "enjoyed",
    "prefer", "prefers", "preferred", "feel", "feels", "felt", "feeling", "think", "thinks",
    "thought", "believe", "believes", "because", "since", "makes", "making", "great", "awesome",
    "nice", "better",
}


@dataclass
class Match:
    term: str
    node: Dict[str, Any]
    via: str          # "alias" | "index"
    score: float


@dataclass
class Resolution:
    terms: List[str] = field(default_factory=list)
    matched: List[Match] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for m in self.matched:
            if m.node["id"] not in seen:
                seen.add(m.node["id"])
                out.append(m.node)
        return out


def heuristic_entities(query: str, max_entities: int = 3) -> List[str]:
    """Contiguous runs of non-stop-words in order of appearance (the first is treated as
    the primary topic). 'what is henna?' -> ['henna']."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", query or "")
    runs: List[List[str]] = []
    cur: List[str] = []
    for t in tokens:
        if t.lower() in _STOP or len(t) < 3:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        runs.append(cur)
    phrases = [" ".join(r[:4]) for r in runs]
    out: List[str] = []
    for p in phrases:
        if p.lower() not in (o.lower() for o in out):
            out.append(p)
    return out[:max_entities]


def _parse_entities_json(text: str) -> Optional[List[str]]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    ents = data.get("entities") if isinstance(data, dict) else None
    if not isinstance(ents, list):
        return None
    clean = [str(e).strip() for e in ents if isinstance(e, (str, int, float)) and str(e).strip()]
    return [e[:80] for e in clean]


class EntityResolver:
    def __init__(self, vector: VectorMemory, graph: GraphMemory, router: Any = None,
                 cfg: Optional[Dict[str, Any]] = None) -> None:
        self.vector = vector
        self.graph = graph
        self.router = router
        self._cfg = cfg

    @property
    def cfg(self) -> Dict[str, Any]:
        return (self._cfg or cognition_config())["resolver"]

    def extract(self, query: str) -> List[str]:
        max_n = int(self.cfg["max_entities"])
        if self.cfg.get("use_llm", True) and self.router is not None:
            try:
                from core.interfaces import Message
                res = self.router.complete(
                    messages=[
                        Message(role="system", content=(
                            "Extract the 1 to 3 core subject-matter topics or domain concepts the user is asking about or discussing, "
                            "most important first. Use short canonical noun phrases (e.g. 'writing style', 'books'). "
                            "Do NOT extract personal conversational sentiment verbs or emotions (e.g., do NOT extract 'love', "
                            "'like', 'opinion', 'preference'). Focus strictly on the underlying subject matter being discussed. "
                            'Reply with JSON only: {"entities": ["..."]}. If there is no topic, reply {"entities": []}.'
                        )),
                        Message(role="user", content=query[:1000]),
                    ],
                    temperature=0.0,
                    max_tokens=80,
                )
                ents = _parse_entities_json(res.get("content", ""))
                if ents is not None:
                    return ents[:max_n]
            except Exception as e:
                logger.debug(f"EntityResolver: LLM extraction failed ({e}); using heuristic")
        return heuristic_entities(query, max_n)

    def resolve_term(self, term: str) -> Optional[Match]:
        for nid in self.graph.resolve_alias(term):
            node = self.graph.get_node(nid)
            if node and node.get("status") != "pruned":
                return Match(term=term, node=node, via="alias", score=1.0)
        floor = float(self.cfg["index_min_similarity"])
        for hit in self.vector.search_entities(term, n_results=3):
            if hit["score"] < floor:
                break
            node = self.graph.get_node(hit["node_id"])
            if node and node.get("status") != "pruned":
                return Match(term=term, node=node, via="index", score=hit["score"])
        return None

    def resolve(self, query: str, terms: Optional[List[str]] = None) -> Resolution:
        terms = terms if terms is not None else self.extract(query)
        res = Resolution(terms=list(terms))
        for t in terms:
            m = self.resolve_term(t)
            if m:
                res.matched.append(m)
            else:
                res.unmatched.append(t)
        return res
