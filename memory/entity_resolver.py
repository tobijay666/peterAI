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
    # question words without apostrophes ("whats the plot of ...")
    "whats", "hows", "whos", "wheres", "whens", "whys", "thats", "theres", "its",
    "what's", "how's", "who's", "where's", "when's", "why's", "that's", "there's", "it's",
    "i'm", "im", "i'd", "id", "i've", "ive", "you're", "youre", "can't", "won't", "don't", "dont",
    "doesn't", "doesnt", "isn't", "isnt", "aren't", "arent", "didn't", "didnt",
    "advice", "tips", "tip", "guide", "recommend", "recommendation", "recommendations", "suggest",
    "suggestion", "suggestions", "weigh", "decide", "choose", "learn", "understand", "start", "begin",
    "hi", "hello", "hey", "thanks", "thank", "okay", "yes", "yeah", "sure", "lot", "lots", "bit",
}

# Words that say WHAT is being asked about a topic, not the topic itself
# ("the main storyline of X", "a summary of X"). They become the query's aspect.
_ASPECT = {
    "main", "storyline", "story", "plot", "summary", "summarise", "summarize", "synopsis", "overview",
    "theme", "themes", "meaning", "message", "moral", "ending", "history", "origin", "origins",
    "definition", "define", "explanation", "difference", "differences", "example", "examples",
    "facts", "fact", "details", "detail", "info", "information", "basics", "introduction",
    "pros", "cons", "advantages", "disadvantages", "benefits", "risks", "importance", "purpose",
    "key", "points", "point", "idea", "ideas", "analysis", "review", "significance", "background",
    "symbolism", "characters", "character", "setting", "author", "writer", "quick", "short", "brief",
}

# Joiners allowed INSIDE a name: "picture of dorian gray", "lord of the rings"
_JOINERS = {"of", "the", "de", "la", "van", "von"}  # not "and": "henna and saree" are two topics


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
    aspect: str = ""      # what is asked about the topic ("main storyline")
    category: str = ""    # broader field the topic belongs to ("Victorian literature")
    subject: str = ""     # main subject, linked straight to Peter ("Writing", "Investing")

    @property
    def nodes(self) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for m in self.matched:
            if m.node["id"] not in seen:
                seen.add(m.node["id"])
                out.append(m.node)
        return out


def _norm_token(t: str) -> str:
    return t.lower().replace("\u2019", "'")


def heuristic_entities(query: str, max_entities: int = 3) -> List[str]:
    """Best-effort topic phrases without an LLM, primary topic first.

    * quoted text ("The Picture of Dorian Gray") is taken whole
    * question words and contractions ("whats") are dropped
    * aspect words ("main storyline", "summary") separate topics and are not topics
    * "of"/"the" may join two parts of one name ("picture of dorian gray"),
      but "by"/"in"/"about" split ("... by oscar wilde" -> a second topic)
    'what is henna?' -> ['henna']
    """
    query = query or ""
    out: List[str] = []

    def add(p: str) -> None:
        p = " ".join(p.split()).strip(" ,.;:!?-'\"")
        if p.lower().startswith("the ") and len(p) > 4:
            p = p[4:]
        if len(p) >= 3 and p.lower() not in (o.lower() for o in out):
            out.append(p)

    for q in re.findall(r'["\u201c\u2018\']([^"\u201d\u2019\']{3,80})["\u201d\u2019\']', query):
        add(q)
        query = query.replace(q, " , ")

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-]*|[,.;:!?]", query)
    runs: List[List[str]] = []
    cur: List[str] = []
    pending: List[str] = []   # joiners seen after a content word, kept only if content follows
    for t in tokens:
        low = _norm_token(t)
        is_content = (low not in _STOP and low not in _ASPECT and low not in _JOINERS
                      and len(t) >= 3 and not re.fullmatch(r"[,.;:!?]", t))
        if is_content:
            cur.extend(pending)
            pending = []
            cur.append(t)
        elif low in _JOINERS and cur:
            pending.append(t)
        else:
            if cur:
                runs.append(cur)
            cur, pending = [], []
    if cur:
        runs.append(cur)
    for r in runs:
        add(" ".join(r[:6]))
    return out[:max_entities]


def heuristic_aspect(query: str) -> str:
    """The aspect words of a query, in order: 'whats the main storyline of X' -> 'main storyline'."""
    words = [w for w in re.findall(r"[A-Za-z']+", query or "") if _norm_token(w) in _ASPECT]
    return " ".join(words[:4])


def _parse_extraction(text: str) -> Optional[Dict[str, Any]]:
    """Find the JSON object with an "entities" list, even with reasoning text or
    <think> blocks around it. Returns {"entities", "aspect", "category"} or None."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL) or []
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
            continue
        ents = [str(e).strip()[:80] for e in data["entities"]
                if isinstance(e, (str, int, float)) and str(e).strip()]
        ents = [e for e in ents if _norm_token(e) not in _STOP and _norm_token(e) not in _ASPECT]
        aspect = str(data.get("aspect") or "").strip()[:80]
        category = str(data.get("category") or "").strip()[:60]
        subject = _subject_name(str(data.get("subject") or ""))
        return {"entities": ents, "aspect": aspect, "category": category, "subject": subject}
    return None


def _subject_name(text: str) -> str:
    """Normalise a main-subject name so 'investing', 'INVESTING ' and 'Investing.' are one node."""
    words = re.sub(r"[^\w\s&'-]", " ", text).split()[:3]
    return " ".join(w if w.isupper() and len(w) > 1 else w.capitalize() for w in words)


def _parse_entities_json(text: str) -> Optional[List[str]]:
    parsed = _parse_extraction(text)
    return parsed["entities"] if parsed is not None else None


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

    _PROMPT = (
        "Identify what the user's message is about.\n"
        "entities: the 1 to 3 specific things it is about, most important first, written as their full "
        "proper names (a book, film, person, place, organisation, product, technique or subject), e.g. "
        "'The Picture of Dorian Gray', 'Oscar Wilde', 'henna'. Never question words ('whats'), aspects "
        "('main storyline', 'summary', 'history'), feelings or generic words ('picture', 'thing').\n"
        "aspect: what is being asked about the main entity, a few words (e.g. 'plot summary'), or ''.\n"
        "category: the broader field of the main entity in 1-3 words (e.g. 'Victorian literature', "
        "'Sri Lanka financial markets', 'Indian cuisine'), or ''.\n"
        "subject: the main school subject or life area this belongs to, 1-2 words, as broad as possible, "
        "preferably one of: Writing, Literature, Investing, Finance, Business, Science, Technology, "
        "Mathematics, History, Culture, Art, Music, Film, Cooking, Health, Sport, Travel, Politics, "
        "Philosophy, Language; or ''.\n"
        'Reply with JSON only: {"entities": ["..."], "aspect": "...", "category": "...", "subject": "..."}. '
        'If the message has no topic (greetings, small talk), reply '
        '{"entities": [], "aspect": "", "category": "", "subject": ""}.'
    )

    def extract_details(self, query: str) -> Dict[str, Any]:
        """{"entities", "aspect", "category", "via"}. LLM first, heuristic fallback."""
        max_n = int(self.cfg["max_entities"])
        if self.cfg.get("use_llm", True) and self.router is not None:
            try:
                from core.interfaces import Message
                res = self.router.complete(
                    messages=[Message(role="system", content=self._PROMPT),
                              Message(role="user", content=query[:1000])],
                    temperature=0.0,
                    # Reasoning models (e.g. gpt-oss) spend tokens thinking before the
                    # JSON; a small budget leaves the reply empty.
                    max_tokens=int(self.cfg.get("llm_max_tokens", 700)),
                )
                parsed = _parse_extraction(res.get("content", ""))
                if parsed is not None:
                    parsed["entities"] = parsed["entities"][:max_n]
                    parsed["via"] = "llm"
                    return parsed
                logger.warning("EntityResolver: LLM reply had no usable JSON; using heuristic")
            except Exception as e:
                logger.warning(f"EntityResolver: LLM extraction failed ({e}); using heuristic")
        return {"entities": heuristic_entities(query, max_n), "aspect": heuristic_aspect(query),
                "category": "", "subject": "", "via": "heuristic"}

    _SUBJECT_PROMPT = (
        "Name the main school subject or life area this belongs to, 1-2 words, as broad as possible, "
        "preferably one of: Writing, Literature, Investing, Finance, Business, Science, Technology, "
        "Mathematics, History, Culture, Art, Music, Film, Cooking, Health, Sport, Travel, Politics, "
        "Philosophy, Language. Reuse one of these existing subjects if it fits: {existing}. "
        'Reply with JSON only: {{"subject": "..."}}.'
    )

    def subject_for(self, text: str, existing: Optional[List[str]] = None) -> str:
        """Main subject for one topic or question ('Colombo Stock Exchange' -> 'Investing').
        '' when no LLM is available or it gives no usable answer."""
        if not text.strip() or self.router is None or not self.cfg.get("use_llm", True):
            return ""
        try:
            from core.interfaces import Message
            res = self.router.complete(
                messages=[Message(role="system", content=self._SUBJECT_PROMPT.format(
                              existing=", ".join(existing or []) or "none")),
                          Message(role="user", content=text[:400])],
                temperature=0.0, max_tokens=int(self.cfg.get("llm_max_tokens", 700)),
            )
            body = re.sub(r"<think>.*?</think>", " ", res.get("content", "") or "", flags=re.DOTALL)
            for raw in reversed(re.findall(r"\{[^{}]*\}", body)):
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("subject"):
                    return _subject_name(str(data["subject"]))
        except Exception as e:
            logger.debug(f"EntityResolver: subject lookup failed ({e})")
        return ""

    def extract(self, query: str) -> List[str]:
        return self.extract_details(query)["entities"]

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
        aspect, category, subject = "", "", ""
        if terms is None:
            details = self.extract_details(query)
            terms, aspect, category = details["entities"], details["aspect"], details["category"]
            subject = details.get("subject", "")
        res = Resolution(terms=list(terms), aspect=aspect, category=category, subject=subject)
        for t in terms:
            m = self.resolve_term(t)
            if m:
                res.matched.append(m)
            else:
                res.unmatched.append(t)
        return res
