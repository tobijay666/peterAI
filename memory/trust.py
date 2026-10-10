"""Memory trust classification and injection scanning (FR-16).

Trust levels (ordered, higher = more trusted):
    system > user > tool_verified > untrusted_web > quarantined

Fail-closed: unknown or missing trust is treated as untrusted_web.
"""
from __future__ import annotations
import re
from enum import Enum
from typing import List, Optional, Tuple
from core.logger import logger


class TrustLevel(str, Enum):
    SYSTEM = "system"
    USER = "user"
    TOOL_VERIFIED = "tool_verified"
    UNTRUSTED_WEB = "untrusted_web"
    QUARANTINED = "quarantined"

    @classmethod
    def from_str(cls, value: Optional[str]) -> "TrustLevel":
        """Parse a trust level string, defaulting to untrusted_web (fail-closed)."""
        if value is None:
            return cls.UNTRUSTED_WEB
        try:
            return cls(value)
        except ValueError:
            return cls.UNTRUSTED_WEB

    def rank(self) -> int:
        """Numeric rank: higher = more trusted."""
        return {
            self.QUARANTINED: 0,
            self.UNTRUSTED_WEB: 1,
            self.TOOL_VERIFIED: 2,
            self.USER: 3,
            self.SYSTEM: 4,
        }[self]

    def __ge__(self, other: "TrustLevel") -> bool:
        return self.rank() >= other.rank()

    def __gt__(self, other: "TrustLevel") -> bool:
        return self.rank() > other.rank()

    def __le__(self, other: "TrustLevel") -> bool:
        return self.rank() <= other.rank()

    def __lt__(self, other: "TrustLevel") -> bool:
        return self.rank() < other.rank()


# ---------------------------------------------------------------------------
# Injection scanner
# ---------------------------------------------------------------------------

# Patterns that indicate prompt-injection or system-override attempts.
# Each is (pattern_string, description). Case-insensitive by default.
_POISON_PATTERNS: List[Tuple[str, str]] = [
    # Direct override commands
    (r"ignore\s+(previous|all|prior)\s+(instructions?|directives?|rules?|prompts?)", "ignore-instructions"),
    (r"disregard\s+(all|prior|previous)", "disregard-prior"),
    (r"forget\s+everything\s+you\s+were\s+told", "forget-instruction"),
    (r"pretend\s+you\s+are", "pretend-jailbreak"),
    (r"act\s+as\s+(if|though)\s+your\s+system\s+prompt", "system-prompt-override"),
    (r"reveal\s+(all\s+)?secrets?", "secret-extraction"),
    # System/role header injection
    (r"\[\s*SYSTEM\s*\]", "system-bracket"),
    (r"\[\s*INST\s*\]", "inst-bracket"),
    (r"###\s*System\s*:", "markdown-system-header"),
    (r"<\|im_start\|>\s*system", "chatml-system"),
    (r'"role"\s*:\s*"system"', "json-role-system"),
    # Tool-call shaped injection
    (r'"tool_call"\s*:\s*"[^"]+",\s*"args"', "fake-tool-call"),
    (r'"function"\s*:\s*"[^"]+"\s*,\s*"text"\s*:\s*"[^"]*ignore', "fake-function-ignore"),
    # DAN / jailbreak
    (r"\bDAN\b", "dan-jailbreak"),
    (r"developer\s+mode", "dev-mode"),
    (r"disable\s+all\s+guardrails?", "disable-guardrails"),
    # Prompt extraction
    (r"(output|reveal|print|show|repeat)\s+(your|the)\s+system\s+prompt", "prompt-extraction"),
    (r"what\s+(are|is)\s+your\s+instructions?", "instruction-query"),
    # Unicode obfuscation (RTL override, zero-width chars)
    (r"\u202e", "rtl-override"),
    (r"[\u200b\u200c\u200d\ufeff]", "zero-width-char"),
    # Role boundary injection
    (r"HUMAN\s*:\s*ignore\s+previous", "human-ignore-previous"),
    (r"new\s+(instruction|role|directive)\s*:", "new-instruction"),
    (r"exfiltrate", "exfiltrate"),
]

_COMPILED_PATTERNS = [
    (re.compile(p, re.IGNORECASE | re.DOTALL), desc)
    for p, desc in _POISON_PATTERNS
]


def injection_scan(text: str) -> Tuple[bool, Optional[str]]:
    """Scan text for injection patterns.

    Returns:
        (is_poisoned: bool, matched_pattern_description: Optional[str])

    This function never raises (safe to call on arbitrary text).
    """
    try:
        for pattern, desc in _COMPILED_PATTERNS:
            if pattern.search(text):
                logger.warning(f"TrustScanner: Injection pattern detected [{desc}] in text (first 80 chars): {text[:80]!r}")
                return True, desc
        return False, None
    except Exception as e:
        logger.error(f"TrustScanner: Unexpected error scanning text: {e}")
        return False, None  # fail open on scanner error, not fail closed


def classify_namespace(namespace: str, kind: str = "") -> TrustLevel:
    """Assign a default trust level based on namespace and record kind."""
    if namespace in ("peter",):
        return TrustLevel.SYSTEM
    if namespace in ("librarian", "devops", "qa", "filesystem", "voice"):
        return TrustLevel.USER
    if namespace == "researcher":
        # Researcher writes are untrusted_web by default;
        # promoted to tool_verified when 2-origin check passes (FR-2.4).
        return TrustLevel.UNTRUSTED_WEB
    if kind in ("legacy_finding",):
        return TrustLevel.UNTRUSTED_WEB
    return TrustLevel.UNTRUSTED_WEB


def min_trust(levels: List[TrustLevel]) -> TrustLevel:
    """Return the lowest trust level from a list (used for derived records)."""
    if not levels:
        return TrustLevel.UNTRUSTED_WEB
    return min(levels, key=lambda t: t.rank())
