"""SkillStore — procedural memory for Peter AI (FR-13).

Skills are plan hints: trigger text + plan template + track record.
They are stored as 'skill' nodes in the graph DB.
Status lifecycle: candidate -> active -> deprecated.
Deprecated skills are kept for lineage, never matched.
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from core.clock import Clock, SystemClock
from core.logger import logger
from memory.graph_store import GraphMemory


DEFAULT_CONFIG = "config/evolution.yaml"


def _load_skill_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("skills", {})
    except Exception:
        return {}


class SkillStore:
    """Manages skill nodes in the GraphMemory.

    All threshold values are read from config/evolution.yaml on init
    and can be reloaded with reload_config().
    """

    def __init__(
        self,
        graph: GraphMemory,
        config_path: str = DEFAULT_CONFIG,
        clock: Optional[Clock] = None,
    ) -> None:
        self.graph = graph
        self.config_path = config_path
        self._clock = clock or SystemClock()
        self._cfg = _load_skill_config(config_path)

    def reload_config(self) -> None:
        self._cfg = _load_skill_config(self.config_path)

    # ------------------------------------------------------------------ create

    def create_candidate(
        self,
        trigger: str,
        plan_template: List[Dict[str, Any]],
        source_task_ids: List[str],
        created_by: str = "consolidation",
    ) -> str:
        """Create a new candidate skill node. Returns skill_id."""
        skill_id = str(uuid.uuid4())
        data = {
            "trigger": trigger,
            "status": "candidate",
            "version": 1,
            "plan_template": plan_template,
            "source_task_ids": source_task_ids,
            "created_by": created_by,
        }
        self.graph.upsert_node(
            node_id=skill_id,
            kind="skill",
            label=trigger[:80],
            data=data,
            importance=0.5,
        )
        # Link to source tasks — upsert stub task nodes first to satisfy FK constraints
        for task_id in source_task_ids:
            try:
                # Ensure the task node exists (may already exist from real task execution)
                existing = self.graph.get_node(task_id)
                if not existing:
                    self.graph.upsert_node(
                        node_id=task_id,
                        kind="task",
                        label=task_id[:80],
                        data={"stub": True},
                    )
                self.graph.link(task_id, skill_id, "derived_from")
            except Exception as e:
                logger.warning(f"SkillStore: Could not link task {task_id} to skill {skill_id}: {e}")
        logger.info(f"SkillStore: Created candidate skill {skill_id}: {trigger[:60]}")
        return skill_id

    # ------------------------------------------------------------------ match

    def match(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return up to k active skills ordered by relevance.

        Since skills are stored in the graph (not Chroma), we do simple
        substring/keyword matching here. Phase E wires in semantic matching
        via the librarian Chroma namespace.
        Returns list of skill data dicts with 'skill_id' added.
        """
        all_nodes = self.graph.get_nodes_by_kind("skill")
        active = [
            n for n in all_nodes
            if n.get("data", {}).get("status") == "active"
        ]
        # Simple relevance: count query words that appear in trigger
        query_words = set(query.lower().split())
        scored = []
        for node in active:
            trigger = node["data"].get("trigger", "").lower()
            score = sum(1 for w in query_words if w in trigger) / max(len(query_words), 1)
            scored.append((score, node))
        scored.sort(key=lambda x: -x[0])
        result = []
        for score, node in scored[:k]:
            if score <= 0:
                break  # sorted descending, no more matches
            entry = dict(node["data"])
            entry["skill_id"] = node["id"]
            entry["match_score"] = score
            result.append(entry)
        return result

    # ------------------------------------------------------------------ promote / deprecate

    def evaluate_promotion(self, skill_id: str, ledger_stats: Dict[str, Any]) -> bool:
        """Promote a candidate skill to active if thresholds are met.

        ledger_stats: {"uses": int, "successes": int} (from v_skill_stats view).
        Returns True if promotion happened.
        """
        min_suc = self._cfg.get("promote_min_successes", 3)
        min_rate = self._cfg.get("promote_min_success_rate", 0.80)

        uses = ledger_stats.get("uses", 0)
        successes = ledger_stats.get("successes", 0)
        rate = successes / uses if uses > 0 else 0.0

        if successes >= min_suc and rate >= min_rate:
            return self._set_status(skill_id, "candidate", "active")
        return False

    def evaluate_deprecation(
        self,
        skill_id: str,
        ledger_stats: Dict[str, Any],
        consecutive_failures: int = 0,
        bad_ratings: int = 0,
    ) -> bool:
        """Deprecate an active skill if any deprecation condition is met.

        Returns True if deprecation happened.
        """
        window = self._cfg.get("deprecate_window", 5)
        below_rate = self._cfg.get("deprecate_below_rate", 0.50)
        max_consec = self._cfg.get("deprecate_after_consecutive_failures", 2)
        max_bad = self._cfg.get("deprecate_after_bad_ratings", 3)

        uses = min(ledger_stats.get("uses", 0), window)
        successes = ledger_stats.get("successes", 0)
        rate = successes / uses if uses > 0 else 0.0

        should_deprecate = (
            (uses >= window and rate < below_rate)
            or consecutive_failures >= max_consec
            or bad_ratings >= max_bad
        )
        if should_deprecate:
            return self._set_status(skill_id, "active", "deprecated")
        return False

    def supersede(self, old_skill_id: str, new_skill_id: str) -> None:
        """Mark old skill as superseded by new skill. Old skill kept for lineage."""
        self._set_status(old_skill_id, None, "deprecated")  # None = any status
        try:
            self.graph.link(new_skill_id, old_skill_id, "supersedes")
        except Exception as e:
            logger.warning(f"SkillStore.supersede link failed: {e}")

    # ------------------------------------------------------------------ internal

    def _set_status(self, skill_id: str, from_status: Optional[str], to_status: str) -> bool:
        node = self.graph.get_node(skill_id)
        if not node:
            return False
        data = dict(node.get("data", {}))
        if from_status and data.get("status") != from_status:
            return False
        data["status"] = to_status
        self.graph.upsert_node(
            node_id=skill_id,
            kind="skill",
            label=data.get("trigger", "")[:80],
            data=data,
            importance=node.get("importance", 0.5),
            pinned=bool(node.get("pinned", False)),
        )
        logger.info(f"SkillStore: Skill {skill_id} -> {to_status}")
        return True

    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        node = self.graph.get_node(skill_id)
        if not node:
            return None
        entry = dict(node.get("data", {}))
        entry["skill_id"] = skill_id
        return entry

    def list_skills(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        nodes = self.graph.get_nodes_by_kind("skill")
        result = []
        for node in nodes:
            data = node.get("data", {})
            if status and data.get("status") != status:
                continue
            entry = dict(data)
            entry["skill_id"] = node["id"]
            result.append(entry)
        return result
