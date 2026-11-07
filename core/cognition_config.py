"""Loads config/cognition.yaml with built-in defaults.

Every cognitive-memory module reads its thresholds through ``cognition_config()``
so nothing is hard-coded. Missing keys fall back to DEFAULTS; a missing or
broken file falls back entirely (the system still runs).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.logger import logger

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "cognition.yaml"

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "usage": {"half_life_days": 14, "baseline_importance": 0.5, "rollup_factor": 0.9},
    "resolver": {"index_min_similarity": 0.55, "max_entities": 3, "use_llm": True,
                 "llm_max_tokens": 700, "link_category": True},
    "recall": {"top_k": 6, "neighbour_weight": 0.6, "max_neighbours": 3,
               "record_fallback_min": 0.45, "record_fallback_max_collections": 64,
               "tree_walk_min": 0.4},
    "vectors": {"similar_add_min": 0.6, "similar_remove_below": 0.45, "similar_k": 3,
                "record_links_k": 3, "record_link_min": 0.30, "follow_weight": 0.5, "max_follow": 6},
    "sufficiency": {"min_score": 0.30, "min_records": 2, "use_llm_judge": True},
    "research": {
        "enabled": True, "max_queries": 3, "max_pages": 4, "max_seconds": 25,
        "max_claims_per_page": 12, "min_claim_chars": 25,
        "min_result_relevance": 0.5, "min_page_relevance": 0.5,
        "corroboration_similarity": 0.90, "async_in_chat": True,
    },
    "curiosity": {
        "per_turn": 4, "dedupe_similarity": 0.85,
        "weights": {"usage": 0.4, "novelty": 0.3, "gap": 0.2, "recency": 0.1},
        "max_pending": 200,
    },
    "fill": {"thin_records": 3, "member_records": 3, "max_hub_records": 40, "max_per_cycle": 3,
             "research_in_chat": False},
    "sleep": {
        "idle_minutes": 20, "min_hours_between": 6, "max_new_collections": 5,
        "research_items_per_cycle": 3,
        "prune": {"percentile_floor": 0.10, "score_floor": 0.05, "window_days": 30, "grace_days": 7},
        "abstraction": {
            "min_similarity": 0.45, "max_similarity": 0.80,
            "migrate_min_similarity": 0.40, "max_records_per_member": 8,
        },
        "merge": {"label_similarity": 0.92, "centroid_similarity": 0.90, "max_per_cycle": 5},
        "gaps": {"min_degree": 2, "min_records": 3, "min_confidence": 0.5},
    },
    "visual": {
        "layout": "graph",
        "tiers": {"high_percentile": 0.80, "low_percentile": 0.30},
        "high": {"scale_min": 1.6, "scale_max": 2.4, "emissive_min": 1.5, "emissive_max": 3.0,
                 "saturation": 1.0, "opacity": 1.0},
        "moderate": {"scale": 1.0, "emissive": 0.4, "saturation": 0.85, "opacity": 0.95},
        "low": {"scale_min": 0.4, "scale_max": 0.7, "emissive": 0.05, "saturation": 0.25,
                "opacity_min": 0.35, "opacity_max": 0.5},
        "fading": {"scale": 0.3, "emissive": 0.0, "saturation": 0.1, "opacity": 0.25},
    },
}

_cache: Optional[Dict[str, Any]] = None


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cognition_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the YAML file (no caching) merged over DEFAULTS.

    PETER_COGNITION_CONFIG=<path> points at an alternative file (tests, smoke runs).
    """
    env_path = os.environ.get("PETER_COGNITION_CONFIG")
    p = Path(path) if path else (Path(env_path) if env_path else CONFIG_PATH)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raw = {}
    except Exception as e:  # malformed YAML must not take Peter down
        logger.error(f"cognition_config: could not parse {p}: {e}; using defaults")
        raw = {}
    return _merge(DEFAULTS, raw)


def cognition_config(reload: bool = False) -> Dict[str, Any]:
    """Cached config accessor used throughout the code base."""
    global _cache
    if _cache is None or reload:
        _cache = load_cognition_config()
    return _cache


def set_cognition_config(cfg: Optional[Dict[str, Any]]) -> None:
    """Override the cached config (tests). Pass None to reset to the file."""
    global _cache
    _cache = _merge(DEFAULTS, cfg) if cfg is not None else None
