"""Enforces that time.time() is not called directly outside core/clock.py (A2)."""
import ast
import sys
from pathlib import Path
import pytest

SOURCE_DIRS = ["core", "memory", "llm", "tools", "agents", "engine"]
ALLOWED_FILE = "core/clock.py"

def _find_time_time_calls(root: Path):
    violations = []
    for src_dir in SOURCE_DIRS:
        for py_file in (root / src_dir).rglob("*.py"):
            rel = py_file.relative_to(root).as_posix()
            if rel == ALLOWED_FILE:
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Attribute) and func.attr == "time":
                        if isinstance(func.value, ast.Attribute) and func.value.attr == "time":
                            violations.append(f"{rel}:{node.lineno}")
                        elif isinstance(func.value, ast.Name) and func.value.id == "time":
                            violations.append(f"{rel}:{node.lineno}")
    return violations

def test_no_direct_time_dot_time():
    """No module outside core/clock.py may call time.time() directly."""
    root = Path("d:/peterAI")
    violations = _find_time_time_calls(root)
    assert violations == [], (
        f"Found {len(violations)} direct time.time() call(s) outside core/clock.py:\n"
        + "\n".join(violations)
    )
