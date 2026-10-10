"""PolicyTuner — proposes provider/model reorderings from ledger data (FR-18).

Emits proposals only; nothing is applied without Sir Anton's `evolve approve`.
Protected paths are never touched.
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.clock import Clock, SystemClock
from core.logger import logger

PROPOSALS_DIR = Path("data/evolution/proposals")

PROTECTED_PATHS = [
    "config/guardrails.json",
    ".env",
    "tests/",
    "core/guardrails.py",
]


class PolicyTuner:
    """Reads the ledger and emits provider/model reordering proposals.

    FR-18.1: Proposals only; never applied unattended.
    FR-18.2: Protected paths never appear in a proposal.
    """

    def __init__(
        self,
        ledger: Any,  # OutcomeLedger
        config_path: str = "config/evolution.yaml",
        clock: Optional[Clock] = None,
    ) -> None:
        self.ledger = ledger
        self._clock = clock or SystemClock()
        self._cfg = self._load_config(config_path)
        PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("tuner", {})
        except Exception:
            return {}

    def generate_proposals(self) -> List[Dict[str, Any]]:
        """Analyse ledger data and generate provider/model proposals.

        Returns list of proposal dicts. Each proposal is also saved to disk.
        Returns [] if below min_samples threshold or data is insufficient.
        """
        min_samples = self._cfg.get("min_samples", 30)
        proposals = []

        try:
            model_stats = self.ledger.stats(group_by="model")
        except Exception as e:
            logger.error(f"PolicyTuner: ledger stats failed: {e}")
            return []

        # Only analyse models with enough samples
        analysed = [s for s in model_stats if (s.get("total") or 0) >= min_samples]
        if len(analysed) < 2:
            logger.info("PolicyTuner: insufficient data for proposals")
            return []

        # Sort by success rate desc, then avg_latency asc
        analysed.sort(key=lambda x: (-((x.get("successes") or 0) / max(x.get("total") or 1, 1)), x.get("avg_latency_ms") or 9999))

        current_order = [s["key"] for s in model_stats]
        proposed_order = [s["key"] for s in analysed]

        if current_order == proposed_order:
            return []  # No change needed

        # Build proposal
        # FR-18.2: Ensure no protected path appears
        for path in PROTECTED_PATHS:
            if any(path in str(m) for m in proposed_order):
                logger.warning(f"PolicyTuner: protected path detected in proposal, aborting")
                return []

        proposal = {
            "id": str(uuid.uuid4()),
            "created_at": self._clock.now(),
            "type": "model_ladder_reorder",
            "current_order": current_order,
            "proposed_order": proposed_order,
            "reason": (
                f"Based on {sum(s.get('total', 0) for s in analysed)} outcomes: "
                f"proposed order has higher success rate. "
                f"Best model: {proposed_order[0] if proposed_order else 'N/A'}."
            ),
            "status": "pending",  # pending | approved | rejected
        }

        # Save to disk
        proposal_path = PROPOSALS_DIR / f"{proposal['id']}.json"
        try:
            proposal_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"PolicyTuner: failed to save proposal: {e}")

        proposals.append(proposal)
        logger.info(f"PolicyTuner: generated proposal {proposal['id']}")
        return proposals

    def list_proposals(self) -> List[Dict[str, Any]]:
        """List all saved proposals."""
        result = []
        try:
            for f in sorted(PROPOSALS_DIR.glob("*.json")):
                try:
                    result.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
        except Exception:
            pass
        return result

    def approve_proposal(self, proposal_id: str) -> bool:
        """Mark a proposal as approved. Actual application is done by the QA gate."""
        path = PROPOSALS_DIR / f"{proposal_id}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["status"] = "approved"
            data["approved_at"] = self._clock.now()
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logger.info(f"PolicyTuner: proposal {proposal_id} approved")
            return True
        except Exception as e:
            logger.error(f"PolicyTuner: approve failed: {e}")
            return False

    def reject_proposal(self, proposal_id: str) -> bool:
        """Mark a proposal as rejected."""
        path = PROPOSALS_DIR / f"{proposal_id}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))  
            data["status"] = "rejected"
            data["rejected_at"] = self._clock.now()
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"PolicyTuner: reject failed: {e}")
            return False
