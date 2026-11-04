"""Security Guardrail Engine evaluating inputs, tool calls, and LLM outputs."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.interfaces import GuardrailResult
from core.logger import logger


class GuardrailManager:
    """Manages dynamic security rules to prevent security breaches and unsafe operations."""

    def __init__(self, config_path: str = "config/guardrails.json"):
        self.config_path = Path(config_path)
        self.rules: List[Dict[str, Any]] = []
        self.compiled_rules: List[Dict[str, Any]] = []
        self.load_rules()

    def load_rules(self) -> None:
        """Loads and compiles rules from the configuration JSON file."""
        if not self.config_path.exists():
            logger.warning(f"Guardrails config file {self.config_path} not found. Running with default empty rules.")
            self.rules = []
            self.compiled_rules = []
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.rules = data.get("rules", [])
                self.compiled_rules = []
                for rule in self.rules:
                    if rule.get("enabled", True):
                        compiled = rule.copy()
                        compiled["regex"] = re.compile(rule["pattern"])
                        self.compiled_rules.append(compiled)
            logger.info(f"Loaded {len(self.compiled_rules)} active guardrail rules.")
        except Exception as e:
            logger.error(f"Failed to load guardrail rules from {self.config_path}: {e}")

    def reload(self) -> None:
        """Hot-reloads guardrail rules from disk."""
        self.load_rules()

    def check_input(self, user_input: str) -> GuardrailResult:
        """Evaluates user prompt text against guardrail rules."""
        return self._evaluate(target_text=user_input, target_type="input")

    def check_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> GuardrailResult:
        """Evaluates tool name and argument parameters against guardrail rules."""
        args_str = json.dumps(tool_args)
        # Check both full JSON string and explicit path parameters if present
        eval_str = f"tool:{tool_name} args:{args_str}"
        return self._evaluate(target_text=eval_str, target_type="tool_call", tool_name=tool_name)

    def check_output(self, output_text: str) -> GuardrailResult:
        """Evaluates LLM generation output against guardrail rules."""
        return self._evaluate(target_text=output_text, target_type="output")

    def check_memory_write(self, text: str, namespace: str = "", trust: str = "") -> GuardrailResult:
        """Check if a memory write is allowed (FR-16.2).
        Runs the injection scanner; quarantined writes are blocked.
        """
        from memory.trust import injection_scan  # avoid circular import at module level
        try:
            poisoned, desc = injection_scan(text)
            if poisoned:
                return GuardrailResult(
                    allowed=False,
                    rule_id="MEMORY_INJECTION",
                    message=f"Memory write blocked: injection pattern detected [{desc}]",
                    action="block",
                )
        except Exception as e:
            logger.warning(f"check_memory_write scanner error (fail-open): {e}")
        # Also run standard output check for any scrubbing rules
        return self.check_output(text)

    def _evaluate(self, target_text: str, target_type: str, tool_name: Optional[str] = None) -> GuardrailResult:
        for rule in self.compiled_rules:
            applies_to = rule.get("applies_to", [])
            if target_type not in applies_to:
                continue

            regex: re.Pattern = rule["regex"]
            if regex.search(target_text):
                rule_id = rule["id"]
                severity = rule.get("severity", "block")
                description = rule.get("description", "Security rule violation")

                msg = f"Guardrail [{rule_id}] triggered ({severity}): {description}"
                logger.warning(f"GUARDRAIL EVENT: {msg}")

                if severity == "block":
                    return GuardrailResult(allowed=False, rule_id=rule_id, message=msg, action="block")
                elif severity == "warn":
                    return GuardrailResult(allowed=True, rule_id=rule_id, message=msg, action="warn")

        return GuardrailResult(allowed=True, message="Passed", action="allow")

    def list_rules(self) -> List[Dict[str, Any]]:
        return self.rules

    def toggle_rule(self, rule_id: str, enabled: bool) -> bool:
        for rule in self.rules:
            if rule["id"] == rule_id:
                rule["enabled"] = enabled
                self._save_and_reload()
                return True
        return False

    def add_rule(self, rule: Dict[str, Any]) -> None:
        self.rules.append(rule)
        self._save_and_reload()

    def remove_rule(self, rule_id: str) -> bool:
        initial_len = len(self.rules)
        self.rules = [r for r in self.rules if r["id"] != rule_id]
        if len(self.rules) < initial_len:
            self._save_and_reload()
            return True
        return False

    def _save_and_reload(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump({"rules": self.rules}, f, indent=2)
            self.load_rules()
        except Exception as e:
            logger.error(f"Failed to save guardrail rules: {e}")
