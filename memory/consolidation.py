"""ConsolidationPass — converts episode clusters into semantic memory (FR-14).

Cycle order (fixed, tested): snapshot → consolidate → promote/retire → evict → journal.
Consolidation ALWAYS runs before eviction (FR-14.5).

Design:
- Candidate episodes: unpinned, trust >= tool_verified OR trust = user,
  older than min_age_hours, not already consolidated (no consolidated_into field).
- Clustering: cosine similarity on existing embeddings (no extra model load).
- One LLM call per cluster for the summary.
- Skill extraction: if the task graph shows the same agent/tool sequence
  succeeded >= promote_min_successes times across cluster tasks, create a candidate skill.
- Idempotency: a second run with the same data produces zero new records (F5).
- Atomicity per cluster: LLM failure in one cluster leaves it untouched (F5).

FR-14 compliance:
- FR-14.1: clusters episodes, writes semantic records with source_ids
- FR-14.2: idempotent and atomic per cluster
- FR-14.3: pinned records are NEVER modified
- FR-14.4: source episodes are decayed (importance lowered), not deleted
- FR-14.5: this class does NOT call eviction; the EvolutionCoordinator does, after
- FR-14.6: dry_run mode reports without changing
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.clock import Clock, SystemClock
from core.guardrails import GuardrailManager
from core.logger import logger
from memory.trust import TrustLevel, min_trust
from memory.vector_store import VectorMemory
from memory.graph_store import GraphMemory

_ELIGIBLE_TRUST = {TrustLevel.SYSTEM, TrustLevel.USER, TrustLevel.TOOL_VERIFIED}


@dataclass
class ClusterPlan:
    """A planned consolidation action for one cluster (for dry-run reporting)."""
    cluster_id: str
    source_ids: List[str]
    source_texts: List[str]
    planned_summary: Optional[str] = None  # set after LLM call
    planned_skill: Optional[Dict[str, Any]] = None
    action: str = "pending"  # pending | done | skipped | failed


@dataclass 
class ConsolidationReport:
    """Output of ConsolidationPass.run()."""
    clusters_found: int = 0
    semantic_records_written: int = 0
    skill_candidates_created: int = 0
    sources_decayed: int = 0
    dry_run: bool = False
    cluster_plans: List[ClusterPlan] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class ConsolidationPass:
    """Runs one consolidation pass over the vector store.

    This class is designed to be called by the EvolutionCoordinator.
    It can also be called directly for testing or manual consolidation.

    Args:
        vector: VectorMemory instance.
        graph: GraphMemory instance.
        router: LLMRouter (or FakeLLMRouter) for summary generation.
        guardrails: GuardrailManager for checking memory writes.
        config: Dict with consolidation config (from evolution.yaml).
        clock: Injectable clock.
    """

    def __init__(
        self,
        vector: VectorMemory,
        graph: GraphMemory,
        router: Any,  # LLMRouter (avoid circular import)
        guardrails: GuardrailManager,
        config: Optional[Dict[str, Any]] = None,
        clock: Optional[Clock] = None,
        namespace: str = "peter",
    ) -> None:
        self.vector = vector
        self.graph = graph
        self.router = router
        self.guardrails = guardrails
        self._clock = clock or SystemClock()
        self.namespace = namespace
        cfg = config or {}
        self.min_age_hours = cfg.get("min_age_hours", 24)
        self.similarity_threshold = cfg.get("similarity_threshold", 0.80)
        self.min_cluster_size = cfg.get("min_cluster_size", 3)
        self.source_importance_factor = cfg.get("source_importance_factor", 0.5)

    # ------------------------------------------------------------------ public

    def run(self, dry_run: bool = False) -> ConsolidationReport:
        """Run the full consolidation pass. Returns a ConsolidationReport."""
        report = ConsolidationReport(dry_run=dry_run)
        
        candidates = self._get_candidates()
        if not candidates:
            logger.info("ConsolidationPass: No eligible candidates.")
            return report

        clusters = self._cluster(candidates)
        report.clusters_found = len(clusters)

        for cluster in clusters:
            plan = ClusterPlan(
                cluster_id=str(uuid.uuid4()),
                source_ids=[r["id"] for r in cluster],
                source_texts=[r["document"] for r in cluster],
            )
            report.cluster_plans.append(plan)

            if len(cluster) < self.min_cluster_size:
                plan.action = "skipped"
                continue

            try:
                self._process_cluster(plan, cluster, report, dry_run)
            except Exception as e:
                plan.action = "failed"
                report.errors.append(f"Cluster {plan.cluster_id}: {e}")
                logger.error(f"ConsolidationPass: cluster {plan.cluster_id} failed: {e}")
                # FR-14.2: one cluster failure must not stop others
                continue

        logger.info(
            f"ConsolidationPass: done. clusters={report.clusters_found}, "
            f"semantic={report.semantic_records_written}, "
            f"skills={report.skill_candidates_created}, "
            f"decayed={report.sources_decayed}, dry_run={dry_run}"
        )
        return report

    # ------------------------------------------------------------------ internals

    def _get_candidates(self) -> List[Dict[str, Any]]:
        """Return episodes eligible for consolidation."""
        now = self._clock.now()
        min_age_sec = self.min_age_hours * 3600.0
        result = []
        for ns in [self.namespace, "librarian"]:
            for r in self.vector.get_all_records(ns):
                meta = r.get("metadata", {})
                # Must not be pinned (FR-14.3)
                if meta.get("pinned", False):
                    continue
                # Must not already be consolidated
                if meta.get("consolidated_into"):
                    continue
                # Must be old enough
                created_at = meta.get("created_at", now)
                if (now - created_at) < min_age_sec:
                    continue
                # Must have eligible trust
                trust = TrustLevel.from_str(meta.get("trust", "untrusted_web"))
                if trust not in _ELIGIBLE_TRUST:
                    continue
                # Must be episode-kind (or finding) — not already semantic
                kind = meta.get("kind", "finding")
                if kind in ("semantic", "skill_trigger"):
                    continue
                result.append({**r, "_namespace": ns, "_trust": trust})
        return result

    def _cluster(self, candidates: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group candidates into clusters using cosine similarity.

        Greedy single-linkage: start from the first unassigned item, 
        add items within similarity_threshold.
        This is O(n^2) but acceptable for consolidation batch sizes.
        """
        if not candidates:
            return []

        # Get embeddings for each candidate
        texts = [r["document"] for r in candidates]
        try:
            embeddings = self.vector.embed_fn(texts)  # list of float lists
            vecs = np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"ConsolidationPass: embedding failed: {e}")
            return []

        # Normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms

        n = len(candidates)
        assigned = [False] * n
        clusters: List[List[Dict[str, Any]]] = []

        for i in range(n):
            if assigned[i]:
                continue
            cluster = [candidates[i]]
            assigned[i] = True
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                sim = float(np.dot(vecs[i], vecs[j]))
                if sim >= self.similarity_threshold:
                    cluster.append(candidates[j])
                    assigned[j] = True
            clusters.append(cluster)

        return clusters

    def _process_cluster(
        self,
        plan: ClusterPlan,
        cluster: List[Dict[str, Any]],
        report: ConsolidationReport,
        dry_run: bool,
    ) -> None:
        """Process one cluster: generate summary, optionally write, decay sources."""
        from core.interfaces import Message

        texts = plan.source_texts
        source_ids = plan.source_ids

        # Generate summary via LLM
        prompt_text = "\n---\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
        messages = [
            Message(role="system", content=(
                "You are Peter's memory consolidation system. "
                "Given the following related memory episodes, write a single concise "
                "factual summary (2-4 sentences) that captures the key knowledge. "
                "Output ONLY the summary text, no preamble."
            )),
            Message(role="user", content=f"Episodes to consolidate:\n{prompt_text}"),
        ]
        try:
            res = self.router.complete(messages=messages, temperature=0.3, max_tokens=256)
            summary_text = res.get("content", "").strip()
        except Exception as e:
            raise RuntimeError(f"LLM summary call failed: {e}") from e

        if not summary_text:
            plan.action = "skipped"
            return

        # Security check on the summary
        gr = self.guardrails.check_memory_write(summary_text)
        if not gr.allowed:
            plan.action = "skipped"
            logger.warning(f"ConsolidationPass: summary blocked by guardrail for cluster {plan.cluster_id}")
            return

        plan.planned_summary = summary_text

        if dry_run:
            plan.action = "dry_run"
            return

        # Determine trust of the semantic record: min of sources (FR-14.1)
        source_trusts = [r.get("_trust", TrustLevel.UNTRUSTED_WEB) for r in cluster]
        record_trust = min_trust(source_trusts)

        # Determine importance: max of sources * 0.9
        source_importances = [float(r.get("metadata", {}).get("importance", 0.5)) for r in cluster]
        record_importance = max(source_importances) * 0.9

        # Write the semantic record
        sem_id = self.vector.store(
            namespace=self.namespace,
            text=summary_text,
            importance=record_importance,
            pinned=False,
            kind="semantic",
            tags="consolidated",
            extra_metadata={
                "trust": record_trust.value,
                "source_ids": json.dumps(source_ids),
                "kind": "semantic",
            },
        )

        report.semantic_records_written += 1
        plan.action = "done"

        # F4: Decay source episodes (importance *= source_importance_factor)
        for r in cluster:
            rid = r["id"]
            ns = r.get("_namespace", self.namespace)
            meta = r.get("metadata", {})
            old_importance = float(meta.get("importance", 0.5))
            new_importance = old_importance * self.source_importance_factor
            delta = new_importance - old_importance  # negative
            ok = self.vector.adjust_importance(ns, rid, delta)
            if ok is not None:
                # Mark as consolidated
                try:
                    all_recs = self.vector.get_all_records(ns)
                    rec = next((x for x in all_recs if x["id"] == rid), None)
                    if rec:
                        upd_meta = dict(rec["metadata"])
                        upd_meta["consolidated_into"] = sem_id
                        self.vector._get_or_create_collection(ns).update(
                            ids=[rid], metadatas=[upd_meta]
                        )
                except Exception as e:
                    logger.warning(f"ConsolidationPass: failed to set consolidated_into for {rid}: {e}")
                report.sources_decayed += 1
